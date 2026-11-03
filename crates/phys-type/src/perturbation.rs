//! Linear perturbations of `N` fields evolved with their background.

use core::ops::{Add, Mul};

use num_complex::Complex64;

use crate::integrate::{AdaptiveSystem, ErrorTolerances, System, component_error_ratio};
use crate::{
    ConformalTime, CovariantMultifieldSystem, CurvedFieldSpaceGeometry, Dimensionless,
    KinematicBasisError, M0, M1, M2, M3, M3Over2, MHalf, MMinusHalf, Mass, MultifieldState,
    MultifieldStateDelta, MultifieldStateDerivative, Quantity, ScaleFactor, Time,
    TwiceDifferentiableMultifieldPotential,
};

/// Rescaled Fourier mode amplitude `v^I_A = a U^I_A`, with dimension fixed by
/// `P_Q = k^3 |U|^2 / (2 pi^2)`.
///
/// The rescaling is dimensionless, so this is also the dimension of the
/// unrescaled `U^I_A`. Only the rescaled variable is ever integrated.
pub type ModeAmplitude<V = Complex64> = Quantity<MMinusHalf, V>;
/// Covariant time derivative `D_t v^I_A`.
pub type ModeVelocity<V = Complex64> = Quantity<MHalf, V>;
/// Covariant second derivative of a rescaled mode amplitude.
pub type ModeAcceleration<V = Complex64> = Quantity<M3Over2, V>;
/// A dimensionless complex `N x N` diagnostic matrix.
pub type ComplexMatrix<const N: usize> = [[Complex64; N]; N];

/// Converts a dimensioned complex value to a real magnitude for error norms.
///
/// The caller supplies an absolute tolerance with the same dimension before
/// combining this magnitude into a dimensionless normalized error.
pub trait ComplexMagnitude {
    /// Returns the complex modulus of the stored numerical value.
    fn complex_magnitude(&self) -> f64;
}

impl<D> ComplexMagnitude for Quantity<D, Complex64> {
    fn complex_magnitude(&self) -> f64 {
        self.value().norm()
    }
}

/// The dimensionless normalized error of one complex component,
/// `|dz| / (a_tol + r_tol * reference)`.
///
/// The absolute tolerance and the reference carry the component's own
/// dimension, so a tolerance meant for a mode amplitude cannot be applied to a
/// mode velocity. The relative tolerance and the returned error are
/// dimensionless, and the modulus keeps a single acceptance decision for the
/// real and imaginary parts together.
///
/// `reference` is the magnitude the component is measured against, and for a
/// mode matrix it is the largest modulus in the matrix rather than the
/// component's own. An individual component may sit exactly at zero — an
/// axis-aligned trajectory puts exact zeros off the diagonal — while the mode
/// as a whole does not, and normalizing each component by itself would let an
/// arbitrarily small one dominate the norm.
pub fn normalized_mode_error<D>(
    error: Quantity<D, Complex64>,
    reference: Quantity<D, f64>,
    absolute_tolerance: Quantity<D, f64>,
    relative_tolerance: f64,
) -> f64 {
    let scale = absolute_tolerance.into_value() + relative_tolerance * reference.into_value().abs();
    if scale == 0.0 {
        // A vanished mode under pure relative control: an exact zero error is
        // fine, anything else is unbounded.
        return if error.complex_magnitude() == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }

    error.complex_magnitude() / scale
}

/// The largest modulus in a mode matrix over both of two states.
fn mode_matrix_scale<D: Copy, const N: usize>(
    current: &[[Quantity<D, Complex64>; N]; N],
    next: &[[Quantity<D, Complex64>; N]; N],
) -> Quantity<D, f64> {
    let largest = current
        .iter()
        .chain(next.iter())
        .flatten()
        .map(ComplexMagnitude::complex_magnitude)
        .fold(0.0, f64::max);

    Quantity::new(largest)
}

/// The two independent complex two-field mode functions, rescaled by the
/// scale factor.
///
/// The integrated variable is `v^I_A = a U^I_A`. Inside the horizon `U`
/// carries a factor `1 / (a sqrt(2k))` and so decays by many orders of
/// magnitude before a mode exits, which forces the absolute error tolerance
/// far below anything the background needs. The rescaled amplitude instead
/// oscillates at fixed size, and its Hubble drag drops from `3H` to `H`.
///
/// Recovering `U` divides by `a`, which
/// [`crate::spectra::mode_covariance`] does once, at the one place where the
/// spectra are formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Modes<const N: usize> {
    /// `v^I_A = a U^I_A`; the first index is field space and the second
    /// labels the independent vacuum oscillator.
    pub rescaled_amplitude: [[ModeAmplitude; N]; N],
    /// `D_t v^I_A` with the same index order.
    pub rescaled_velocity: [[ModeVelocity; N]; N],
}

/// Background and mode matrix for one comoving wavenumber.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerturbationState<const N: usize> {
    /// Homogeneous two-field background.
    pub background: MultifieldState<N>,
    /// Complex `2 x 2` mode matrix and its covariant velocity.
    pub modes: Modes<N>,
}

/// Time derivative of [`PerturbationState<N>`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerturbationDerivative<const N: usize> {
    /// Background derivative.
    pub background: MultifieldStateDerivative<N>,
    /// Coordinate time derivative of the mode matrix.
    pub amplitude: [[ModeVelocity; N]; N],
    /// Coordinate time derivative of the covariant mode velocity.
    pub velocity: [[ModeAcceleration; N]; N],
}

/// Finite change of a background-plus-modes state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerturbationDelta<const N: usize> {
    /// Background change.
    pub background: MultifieldStateDelta<N>,
    /// Change in mode amplitudes.
    pub amplitude: [[ModeAmplitude; N]; N],
    /// Change in covariant mode velocities.
    pub velocity: [[ModeVelocity; N]; N],
}

/// Two-field background variables evolved in conformal time.
///
/// The scale factor is stored as `ln a`.  This removes the exponentially large
/// component from the adaptive state while still allowing an ordinary
/// [`MultifieldState`] snapshot to be reconstructed wherever physical
/// diagnostics need `a`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalBackgroundState<const N: usize> {
    /// Scalar field values `phi^I`.
    pub phi: [crate::Field; N],
    /// Cosmic-time field velocities `d phi^I / dt`.
    pub pi: [crate::FieldVelocity; N],
    /// Natural logarithm of the scale factor.
    pub log_scale_factor: Quantity<M0, f64>,
}

impl<const N: usize> ConformalBackgroundState<N> {
    /// Converts an ordinary background snapshot to the conformal state.
    pub fn from_cosmic(state: MultifieldState<N>) -> Self {
        Self {
            phi: state.phi,
            pi: state.pi,
            log_scale_factor: Quantity::new(state.a.into_value().ln()),
        }
    }

    /// Reconstructs an ordinary background snapshot.
    pub fn cosmic(self) -> MultifieldState<N> {
        MultifieldState::new(
            self.phi,
            self.pi,
            ScaleFactor::new(self.log_scale_factor.into_value().exp()),
        )
    }

    /// E-folds elapsed from an ordinary initial background.
    pub fn e_folds(&self, initial: &MultifieldState<N>) -> f64 {
        self.log_scale_factor.into_value() - initial.a.into_value().ln()
    }
}

/// Background and rescaled modes evolved with conformal time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalPerturbationState<const N: usize> {
    /// Background with `ln a` as its expansion variable.
    pub background: ConformalBackgroundState<N>,
    /// `v = a U` and its covariant conformal-time derivative `D_eta v`.
    pub modes: Modes<N>,
}

impl<const N: usize> ConformalPerturbationState<N> {
    /// Returns the equivalent cosmic-time state for spectra and diagnostics.
    pub fn cosmic(self) -> PerturbationState<N> {
        let background = self.background.cosmic();
        let inverse_scale_factor = 1.0 / background.a.into_value();

        PerturbationState {
            background,
            modes: Modes {
                rescaled_amplitude: self.modes.rescaled_amplitude,
                // D_eta v = a D_t v.
                rescaled_velocity: self
                    .modes
                    .rescaled_velocity
                    .map(|row| row.map(|entry| entry * inverse_scale_factor)),
            },
        }
    }
}

/// Conformal-time derivative of a background-plus-modes state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalPerturbationDerivative<const N: usize> {
    /// `d phi / d eta`.
    pub phi: [Quantity<M2, f64>; N],
    /// `d pi / d eta`.
    pub pi: [Quantity<M3, f64>; N],
    /// `d ln(a) / d eta = a H`.
    pub log_scale_factor: Quantity<M1, f64>,
    /// Coordinate derivative of `v`.
    pub amplitude: [[ModeVelocity; N]; N],
    /// Coordinate derivative of `D_eta v`.
    pub velocity: [[ModeAcceleration; N]; N],
}

/// Finite conformal-time change of a background-plus-modes state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalPerturbationDelta<const N: usize> {
    /// Changes in the scalar fields.
    pub phi: [crate::Field; N],
    /// Changes in the cosmic-time field velocities.
    pub pi: [crate::FieldVelocity; N],
    /// Change in `ln a`.
    pub log_scale_factor: Quantity<M0, f64>,
    /// Change in `v`.
    pub amplitude: [[ModeAmplitude; N]; N],
    /// Change in `D_eta v`.
    pub velocity: [[ModeVelocity; N]; N],
}

impl<const N: usize> Add<ConformalPerturbationDelta<N>> for ConformalPerturbationState<N> {
    type Output = Self;

    fn add(self, rhs: ConformalPerturbationDelta<N>) -> Self::Output {
        Self {
            background: ConformalBackgroundState {
                phi: core::array::from_fn(|index| self.background.phi[index] + rhs.phi[index]),
                pi: core::array::from_fn(|index| self.background.pi[index] + rhs.pi[index]),
                log_scale_factor: self.background.log_scale_factor + rhs.log_scale_factor,
            },
            modes: Modes {
                rescaled_amplitude: add_matrices(self.modes.rescaled_amplitude, rhs.amplitude),
                rescaled_velocity: add_matrices(self.modes.rescaled_velocity, rhs.velocity),
            },
        }
    }
}

impl<const N: usize> Add for ConformalPerturbationDerivative<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            phi: core::array::from_fn(|index| self.phi[index] + rhs.phi[index]),
            pi: core::array::from_fn(|index| self.pi[index] + rhs.pi[index]),
            log_scale_factor: self.log_scale_factor + rhs.log_scale_factor,
            amplitude: add_matrices(self.amplitude, rhs.amplitude),
            velocity: add_matrices(self.velocity, rhs.velocity),
        }
    }
}

impl<const N: usize> Mul<f64> for ConformalPerturbationDerivative<N> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            phi: self.phi.map(|entry| entry * rhs),
            pi: self.pi.map(|entry| entry * rhs),
            log_scale_factor: self.log_scale_factor * rhs,
            amplitude: scale_matrix(self.amplitude, rhs),
            velocity: scale_matrix(self.velocity, rhs),
        }
    }
}

impl<const N: usize> Mul<ConformalTime> for ConformalPerturbationDerivative<N> {
    type Output = ConformalPerturbationDelta<N>;

    fn mul(self, rhs: ConformalTime) -> Self::Output {
        let interval = rhs.as_time();
        ConformalPerturbationDelta {
            phi: self.phi.map(|entry| entry * interval),
            pi: self.pi.map(|entry| entry * interval),
            log_scale_factor: self.log_scale_factor * interval,
            amplitude: self.amplitude.map(|row| row.map(|entry| entry * interval)),
            velocity: self.velocity.map(|row| row.map(|entry| entry * interval)),
        }
    }
}

impl<const N: usize> Add<PerturbationDelta<N>> for PerturbationState<N> {
    type Output = Self;

    fn add(self, rhs: PerturbationDelta<N>) -> Self::Output {
        Self {
            background: self.background + rhs.background,
            modes: Modes {
                rescaled_amplitude: add_matrices(self.modes.rescaled_amplitude, rhs.amplitude),
                rescaled_velocity: add_matrices(self.modes.rescaled_velocity, rhs.velocity),
            },
        }
    }
}

impl<const N: usize> Add for PerturbationDerivative<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            background: self.background + rhs.background,
            amplitude: add_matrices(self.amplitude, rhs.amplitude),
            velocity: add_matrices(self.velocity, rhs.velocity),
        }
    }
}

impl<const N: usize> Mul<f64> for PerturbationDerivative<N> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            background: self.background * rhs,
            amplitude: scale_matrix(self.amplitude, rhs),
            velocity: scale_matrix(self.velocity, rhs),
        }
    }
}

impl<const N: usize> Mul<Time> for PerturbationDerivative<N> {
    type Output = PerturbationDelta<N>;

    fn mul(self, rhs: Time) -> Self::Output {
        PerturbationDelta {
            background: self.background * rhs,
            amplitude: self.amplitude.map(|row| row.map(|entry| entry * rhs)),
            velocity: self.velocity.map(|row| row.map(|entry| entry * rhs)),
        }
    }
}

fn add_matrices<D: Copy, const N: usize>(
    lhs: [[Quantity<D, Complex64>; N]; N],
    rhs: [[Quantity<D, Complex64>; N]; N],
) -> [[Quantity<D, Complex64>; N]; N] {
    core::array::from_fn(|row| core::array::from_fn(|column| lhs[row][column] + rhs[row][column]))
}

fn scale_matrix<D, const N: usize>(
    matrix: [[Quantity<D, Complex64>; N]; N],
    coefficient: f64,
) -> [[Quantity<D, Complex64>; N]; N] {
    matrix.map(|row| row.map(|entry| entry * coefficient))
}

/// An orthonormal frame `frames[a][I]` whose first vector is the background
/// tangent, for the Bunch-Davies vacuum.
///
/// For two fields the second vector is the positively oriented normal,
/// computed exactly as in [`crate::KinematicBasis2`]. For other `N` the
/// remaining vectors come from Gram-Schmidt on the coordinate basis. The
/// vacuum covariance does not depend on this choice; only the labelling of
/// the independent oscillators does.
fn vacuum_frame<P, G, const N: usize>(
    system: &CovariantMultifieldSystem<P, G, N>,
    state: &MultifieldState<N>,
) -> Result<[[f64; N]; N], KinematicBasisError>
where
    P: crate::MultifieldPotential<N>,
    G: crate::FieldSpaceGeometry<N>,
{
    let metric = system.geometry.metric(&state.phi);
    let g = |i: usize, j: usize| metric.components()[i][j].into_value();
    let kinematics = system.kinematics(state)?;
    let tangent: [f64; N] = kinematics.tangent.components().map(Quantity::into_value);
    let mut frames = [[0.0; N]; N];
    frames[0] = tangent;
    if N == 1 {
        return Ok(frames);
    }
    let lower = |v: &[f64; N]| -> [f64; N] {
        core::array::from_fn(|i| (0..N).map(|j| g(i, j) * v[j]).sum())
    };
    if N == 2 {
        let determinant = g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0);
        if g(0, 0) <= 0.0 || determinant <= 0.0 {
            return Err(KinematicBasisError::NonPositiveMetric);
        }
        let tangent_lower = lower(&tangent);
        let root_determinant = determinant.sqrt();
        frames[1][0] = -tangent_lower[1] / root_determinant;
        frames[1][1] = tangent_lower[0] / root_determinant;
        return Ok(frames);
    }
    let mut count = 1;
    for axis in 0..N {
        let mut vector = [0.0; N];
        vector[axis] = 1.0;
        for frame in frames.iter().take(count) {
            let overlap: f64 = lower(frame)
                .iter()
                .zip(vector.iter())
                .map(|(a, b)| a * b)
                .sum();
            for (component, basis) in vector.iter_mut().zip(frame.iter()) {
                *component -= overlap * basis;
            }
        }
        let norm: f64 = lower(&vector)
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        if norm > 1.0e-12 {
            frames[count] = vector.map(|component| component / norm.sqrt());
            count += 1;
            if count == N {
                return Ok(frames);
            }
        }
    }
    Err(KinematicBasisError::NonPositiveMetric)
}

/// Failure to construct an adiabatic-vacuum mode state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModeInitializationError {
    /// The comoving wavenumber must be positive and finite.
    InvalidWavenumber,
    /// The requested mode is not sufficiently far inside the Hubble radius.
    NotDeepInsideHorizon {
        /// Actual `k/(aH)`.
        actual: f64,
        /// Required minimum value.
        required: f64,
    },
    /// The background tangent/normal basis is undefined.
    KinematicBasis(KinematicBasisError),
}

/// A two-field background system plus one comoving perturbation wavenumber.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerturbationSystem<P, G, const N: usize> {
    /// Background dynamics and geometric data.
    pub background: CovariantMultifieldSystem<P, G, N>,
    /// Positive comoving wavenumber.
    pub k: Mass,
}

impl<P, G, const N: usize> PerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    /// Builds a one-wavenumber perturbation system.
    pub const fn new(background: CovariantMultifieldSystem<P, G, N>, k: Mass) -> Self {
        Self { background, k }
    }

    /// Constructs a Bunch-Davies state using the instantaneous adiabatic and
    /// entropy frame, with a common phase chosen to be zero.
    pub fn bunch_davies_state(
        &self,
        background: MultifieldState<N>,
        minimum_depth: f64,
    ) -> Result<PerturbationState<N>, ModeInitializationError> {
        if self.k.into_value() <= 0.0 || !self.k.into_value().is_finite() {
            return Err(ModeInitializationError::InvalidWavenumber);
        }
        let hubble = self.background.hubble(&background);
        let depth = (self.k / (background.a * hubble)).into_value();
        if !minimum_depth.is_finite() || minimum_depth <= 0.0 || depth < minimum_depth {
            return Err(ModeInitializationError::NotDeepInsideHorizon {
                actual: depth,
                required: minimum_depth,
            });
        }
        let frames = vacuum_frame(&self.background, &background)
            .map_err(ModeInitializationError::KinematicBasis)?;
        let normalization = (self.k * 2.0).sqrt();
        // The rescaled vacuum amplitude carries no factor of the scale
        // factor, so it is the same size for every mode: |v| = 1/sqrt(2k).
        let scalar_amplitude: ModeAmplitude<f64> = Dimensionless::new(1.0) / normalization;
        let physical_wavenumber = self.k / background.a;
        // D_t v = a D_t U + a H U = -i (k/a) v: rescaling cancels the Hubble
        // term exactly, leaving a pure oscillation.
        let frequency =
            Quantity::<M1, Complex64>::new(Complex64::new(0.0, -physical_wavenumber.into_value()));
        let rescaled_amplitude = core::array::from_fn(|field| {
            core::array::from_fn(|mode| {
                let value = frames[mode][field] * scalar_amplitude.into_value();
                ModeAmplitude::new(Complex64::new(value, 0.0))
            })
        });
        let rescaled_velocity = rescaled_amplitude.map(|row| row.map(|entry| entry * frequency));

        Ok(PerturbationState {
            background,
            modes: Modes {
                rescaled_amplitude,
                rescaled_velocity,
            },
        })
    }

    /// Returns `i a^3 (U^dagger P - P^dagger U)`, whose canonical value is
    /// the identity matrix.
    pub fn wronskian(&self, state: &PerturbationState<N>) -> ComplexMatrix<N> {
        let metric = self.background.geometry.metric(&state.background.phi);
        // With v = a U the canonical Wronskian keeps one power of a rather
        // than three, because two are cancelled by the rescaling.
        let scale_factor = state.background.a.into_value();
        core::array::from_fn(|left_mode| {
            core::array::from_fn(|right_mode| {
                let difference = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(Complex64::new(0.0, 0.0), |sum, (first, second)| {
                        let g = *metric.components()[first][second].value();
                        let u_left = state.modes.rescaled_amplitude[first][left_mode].into_value();
                        let p_left = state.modes.rescaled_velocity[first][left_mode].into_value();
                        let u_right =
                            state.modes.rescaled_amplitude[second][right_mode].into_value();
                        let p_right =
                            state.modes.rescaled_velocity[second][right_mode].into_value();
                        sum + g * (u_left.conj() * p_right - p_left.conj() * u_right)
                    });

                Complex64::new(0.0, scale_factor) * difference
            })
        })
    }
}

impl<P, G, const N: usize> System for PerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    type IndependentVariable = Time;
    type State = PerturbationState<N>;
    type Delta = PerturbationDelta<N>;
    type Derivative = PerturbationDerivative<N>;

    fn derivative(&self, time: Time, state: &Self::State) -> Self::Derivative {
        let background_derivative = self.background.derivative(time, &state.background);
        let hubble = self.background.hubble(&state.background);
        let connection = self.background.geometry.christoffel(&state.background.phi);
        let mass = self.background.perturbation_mass_matrix(&state.background);
        let physical_wavenumber = self.k / state.background.a;
        let wave_number_squared = physical_wavenumber * physical_wavenumber;
        // Rescaling U by a turns `D_t^2 U + 3H D_t U + (M + k^2/a^2) U = 0`
        // into `D_t^2 v + H D_t v + (M + k^2/a^2 - (2 - epsilon_H) H^2) v = 0`.
        // The subtracted term is `a''/a` written in cosmic time.
        let expansion = hubble * hubble * (2.0 - self.background.epsilon_h(&state.background));
        let amplitude = core::array::from_fn(|upper| {
            core::array::from_fn(|mode| {
                let connection_term = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(
                        ModeVelocity::new(Complex64::new(0.0, 0.0)),
                        |sum, (first, second)| {
                            sum + connection.components()[upper][first][second]
                                * state.background.pi[first]
                                * state.modes.rescaled_amplitude[second][mode]
                        },
                    );
                state.modes.rescaled_velocity[upper][mode] - connection_term
            })
        });
        let velocity = core::array::from_fn(|upper| {
            core::array::from_fn(|mode| {
                let restoring = (0..N).fold(
                    ModeAcceleration::new(Complex64::new(0.0, 0.0)),
                    |sum, lower| {
                        let entry = if upper == lower {
                            mass.components()[upper][lower] + wave_number_squared - expansion
                        } else {
                            mass.components()[upper][lower]
                        };
                        sum + entry * state.modes.rescaled_amplitude[lower][mode]
                    },
                );
                let connection_term = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(
                        ModeAcceleration::new(Complex64::new(0.0, 0.0)),
                        |sum, (first, second)| {
                            sum + connection.components()[upper][first][second]
                                * state.background.pi[first]
                                * state.modes.rescaled_velocity[second][mode]
                        },
                    );

                state.modes.rescaled_velocity[upper][mode] * hubble * -1.0
                    - restoring
                    - connection_term
            })
        });

        PerturbationDerivative {
            background: background_derivative,
            amplitude,
            velocity,
        }
    }
}

/// A two-field perturbation system written in conformal time.
///
/// It evolves `ln a` rather than `a`, and stores `D_eta v`.  The mode equation
/// is therefore
/// `D_eta^2 v + [k^2 + a^2 C - a''/a] v = 0`, with no Hubble-drag term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalPerturbationSystem<P, G, const N: usize> {
    /// Background dynamics and geometric data.
    pub background: CovariantMultifieldSystem<P, G, N>,
    /// Positive comoving wavenumber.
    pub k: Mass,
}

impl<P, G, const N: usize> ConformalPerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N> + Clone,
    G: CurvedFieldSpaceGeometry<N> + Clone,
{
    /// Builds a one-wavenumber conformal-time system.
    pub const fn new(background: CovariantMultifieldSystem<P, G, N>, k: Mass) -> Self {
        Self { background, k }
    }

    /// Constructs the same Bunch-Davies vacuum as the cosmic-time system,
    /// converting `D_t v` to `D_eta v = a D_t v` at the boundary.
    pub fn bunch_davies_state(
        &self,
        background: MultifieldState<N>,
        minimum_depth: f64,
    ) -> Result<ConformalPerturbationState<N>, ModeInitializationError> {
        let cosmic = PerturbationSystem::new(self.background.clone(), self.k)
            .bunch_davies_state(background, minimum_depth)?;
        let scale_factor = cosmic.background.a.into_value();

        Ok(ConformalPerturbationState {
            background: ConformalBackgroundState::from_cosmic(cosmic.background),
            modes: Modes {
                rescaled_amplitude: cosmic.modes.rescaled_amplitude,
                rescaled_velocity: cosmic
                    .modes
                    .rescaled_velocity
                    .map(|row| row.map(|entry| entry * scale_factor)),
            },
        })
    }

    /// Canonical Wronskian, expressed directly with `D_eta v`.
    pub fn wronskian(&self, state: &ConformalPerturbationState<N>) -> ComplexMatrix<N> {
        PerturbationSystem::new(self.background.clone(), self.k).wronskian(&state.cosmic())
    }
}

impl<P, G, const N: usize> System for ConformalPerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    type IndependentVariable = ConformalTime;
    type State = ConformalPerturbationState<N>;
    type Delta = ConformalPerturbationDelta<N>;
    type Derivative = ConformalPerturbationDerivative<N>;

    fn derivative(&self, _conformal_time: ConformalTime, state: &Self::State) -> Self::Derivative {
        let background = state.background.cosmic();
        let scale_factor = background.a;
        let cosmic_derivative = self.background.derivative(Time::new(0.0), &background);
        let hubble = self.background.hubble(&background);
        let epsilon_h = self.background.epsilon_h(&background);
        let connection = self.background.geometry.christoffel(&background.phi);
        let mass = self.background.perturbation_mass_matrix(&background);
        let conformal_field_velocity = background.pi.map(|entry| entry * scale_factor);
        let scale_factor_squared = scale_factor * scale_factor;
        let expansion = hubble * hubble * scale_factor_squared * (2.0 - epsilon_h);
        let wave_number_squared = self.k * self.k;

        let amplitude = core::array::from_fn(|upper| {
            core::array::from_fn(|mode| {
                let connection_term = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(
                        ModeVelocity::new(Complex64::new(0.0, 0.0)),
                        |sum, (first, second)| {
                            sum + connection.components()[upper][first][second]
                                * conformal_field_velocity[first]
                                * state.modes.rescaled_amplitude[second][mode]
                        },
                    );
                state.modes.rescaled_velocity[upper][mode] - connection_term
            })
        });
        let velocity = core::array::from_fn(|upper| {
            core::array::from_fn(|mode| {
                let restoring = (0..N).fold(
                    ModeAcceleration::new(Complex64::new(0.0, 0.0)),
                    |sum, lower| {
                        let entry = if upper == lower {
                            mass.components()[upper][lower] * scale_factor_squared
                                + wave_number_squared
                                - expansion
                        } else {
                            mass.components()[upper][lower] * scale_factor_squared
                        };
                        sum + entry * state.modes.rescaled_amplitude[lower][mode]
                    },
                );
                let connection_term = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(
                        ModeAcceleration::new(Complex64::new(0.0, 0.0)),
                        |sum, (first, second)| {
                            sum + connection.components()[upper][first][second]
                                * conformal_field_velocity[first]
                                * state.modes.rescaled_velocity[second][mode]
                        },
                    );

                -restoring - connection_term
            })
        });

        ConformalPerturbationDerivative {
            phi: cosmic_derivative.dphi_dt.map(|entry| entry * scale_factor),
            pi: cosmic_derivative.dpi_dt.map(|entry| entry * scale_factor),
            log_scale_factor: scale_factor * hubble,
            amplitude,
            velocity,
        }
    }
}

/// Absolute tolerances for the conformal-time state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalBackgroundTolerance<const N: usize> {
    /// Absolute field tolerances.
    pub phi: [crate::Field<f64>; N],
    /// Absolute cosmic-velocity tolerances.
    pub pi: [crate::FieldVelocity<f64>; N],
    /// Absolute tolerance for `ln a`.
    pub log_scale_factor: Quantity<M0, f64>,
}

/// Absolute tolerances for the conformal-time state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalPerturbationTolerance<const N: usize> {
    /// Tolerances for `phi`, cosmic `pi`, and `ln a`.
    pub background: ConformalBackgroundTolerance<N>,
    /// Absolute tolerance for `v`.
    pub rescaled_amplitude: ModeAmplitude<f64>,
    /// Absolute tolerance for `D_eta v`.
    pub rescaled_velocity: ModeVelocity<f64>,
}

impl<const N: usize> Default for ConformalBackgroundTolerance<N> {
    fn default() -> Self {
        Self {
            phi: [crate::Field::new(0.0); N],
            pi: [crate::FieldVelocity::new(0.0); N],
            log_scale_factor: Quantity::new(0.0),
        }
    }
}

impl<const N: usize> Default for ConformalPerturbationTolerance<N> {
    fn default() -> Self {
        Self {
            background: ConformalBackgroundTolerance::default(),
            rescaled_amplitude: ModeAmplitude::new(0.0),
            rescaled_velocity: ModeVelocity::new(0.0),
        }
    }
}

impl<const N: usize> ConformalPerturbationTolerance<N> {
    /// Bounds the real background components and leaves modes under pure
    /// relative control.
    pub fn background_in_planck_units(value: f64) -> Self {
        Self {
            background: ConformalBackgroundTolerance {
                phi: [crate::Field::new(value); N],
                pi: [crate::FieldVelocity::new(value); N],
                log_scale_factor: Quantity::new(value),
            },
            ..Self::default()
        }
    }
}

impl<P, G, const N: usize> AdaptiveSystem for ConformalPerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    type AbsoluteTolerance = ConformalPerturbationTolerance<N>;

    fn error_ratio(
        &self,
        current: &Self::State,
        next: &Self::State,
        error: &Self::Delta,
        tolerances: &ErrorTolerances<Self::AbsoluteTolerance>,
    ) -> f64 {
        let absolute = tolerances.absolute;
        let relative = tolerances.relative;
        let fields = (0..N).map(|index| {
            component_error_ratio(
                error.phi[index],
                current.background.phi[index],
                next.background.phi[index],
                absolute.background.phi[index],
                relative,
            )
        });
        let velocities = (0..N).map(|index| {
            component_error_ratio(
                error.pi[index],
                current.background.pi[index],
                next.background.pi[index],
                absolute.background.pi[index],
                relative,
            )
        });
        let log_scale_factor = component_error_ratio(
            error.log_scale_factor,
            current.background.log_scale_factor,
            next.background.log_scale_factor,
            absolute.background.log_scale_factor,
            relative,
        );
        let background = fields.chain(velocities).fold(log_scale_factor, f64::max);
        let amplitude_scale = mode_matrix_scale(
            &current.modes.rescaled_amplitude,
            &next.modes.rescaled_amplitude,
        );
        let velocity_scale = mode_matrix_scale(
            &current.modes.rescaled_velocity,
            &next.modes.rescaled_velocity,
        );

        (0..N)
            .flat_map(|field| (0..N).map(move |mode| (field, mode)))
            .map(|(field, mode)| {
                normalized_mode_error(
                    error.amplitude[field][mode],
                    amplitude_scale,
                    absolute.rescaled_amplitude,
                    relative,
                )
                .max(normalized_mode_error(
                    error.velocity[field][mode],
                    velocity_scale,
                    absolute.rescaled_velocity,
                    relative,
                ))
            })
            .fold(background, f64::max)
    }
}

/// Absolute tolerances for a background-plus-modes state.
///
/// The mode entries default to zero, which is not an oversight. A mode
/// component is complex and is normalized by its modulus, and the modulus of
/// a physical mode never vanishes, so no floor is needed to keep the ratio
/// finite. Any floor placed there instead becomes the accuracy ceiling for
/// the modes, which is how the run previously lost its convergence in
/// `x_init` without failing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerturbationTolerance<const N: usize> {
    /// Absolute tolerances for the background components.
    pub background: MultifieldStateDelta<N>,
    /// Absolute tolerance for every rescaled mode amplitude.
    pub rescaled_amplitude: ModeAmplitude<f64>,
    /// Absolute tolerance for every rescaled mode velocity.
    pub rescaled_velocity: ModeVelocity<f64>,
}

impl<const N: usize> PerturbationTolerance<N> {
    /// Bounds the background in Planck units and leaves the modes under pure
    /// relative control.
    pub fn background_in_planck_units(value: f64) -> Self {
        Self {
            background: MultifieldStateDelta::from_planck_units(value),
            ..Self::default()
        }
    }
}

impl<P, G, const N: usize> AdaptiveSystem for PerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    type AbsoluteTolerance = PerturbationTolerance<N>;

    /// Combines the background norm with the complex mode norms.
    ///
    /// Each mode component contributes the modulus of its whole complex
    /// error, so the real and imaginary parts are never accepted separately.
    fn error_ratio(
        &self,
        current: &Self::State,
        next: &Self::State,
        error: &Self::Delta,
        tolerances: &ErrorTolerances<PerturbationTolerance<N>>,
    ) -> f64 {
        let background = self.background.error_ratio(
            &current.background,
            &next.background,
            &error.background,
            &ErrorTolerances::new(tolerances.absolute.background, tolerances.relative),
        );

        // One scale per mode matrix, not per component: see
        // `normalized_mode_error`.
        let amplitude_scale = mode_matrix_scale(
            &current.modes.rescaled_amplitude,
            &next.modes.rescaled_amplitude,
        );
        let velocity_scale = mode_matrix_scale(
            &current.modes.rescaled_velocity,
            &next.modes.rescaled_velocity,
        );

        (0..N)
            .flat_map(|field| (0..N).map(move |mode| (field, mode)))
            .map(|(field, mode)| {
                let amplitude = normalized_mode_error(
                    error.amplitude[field][mode],
                    amplitude_scale,
                    tolerances.absolute.rescaled_amplitude,
                    tolerances.relative,
                );
                let velocity = normalized_mode_error(
                    error.velocity[field][mode],
                    velocity_scale,
                    tolerances.absolute.rescaled_velocity,
                    tolerances.relative,
                );

                amplitude.max(velocity)
            })
            .fold(background, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ErrorTolerances, Field, FieldVelocity, FlatFieldSpace, QuadraticMultifieldPotential,
        ScaleFactor, adaptive_step, rk4_step,
    };

    fn system() -> PerturbationSystem<QuadraticMultifieldPotential<2>, FlatFieldSpace, 2> {
        let background = CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(1.0e-6), Mass::new(1.5e-6)]),
            FlatFieldSpace,
        );
        PerturbationSystem::new(background, Mass::new(1.0e-2))
    }

    fn background() -> MultifieldState<2> {
        MultifieldState::new(
            [Field::new(12.0), Field::new(8.0)],
            [FieldVelocity::new(-3.0e-6), FieldVelocity::new(-4.0e-6)],
            ScaleFactor::new(1.0),
        )
    }

    #[test]
    fn complex_quantity_retains_dimension_and_uses_modulus() {
        let value = ModeAmplitude::new(Complex64::new(3.0, 4.0));
        assert_eq!(value.complex_magnitude(), 5.0);
    }

    #[test]
    fn normalized_error_uses_the_modulus_of_the_whole_complex_component() {
        let error = ModeAmplitude::new(Complex64::new(3.0e-9, 4.0e-9));
        let reference = ModeAmplitude::<f64>::new(2.0);
        let absolute = ModeAmplitude::<f64>::new(0.0);

        // |error| = 5e-9 against a reference of 2.
        let ratio = normalized_mode_error(error, reference, absolute, 1.0e-9);

        assert!((ratio - 2.5).abs() < 1.0e-12);
    }

    #[test]
    fn a_purely_imaginary_error_is_not_ignored() {
        let reference = ModeAmplitude::<f64>::new(1.0);
        let absolute = ModeAmplitude::<f64>::new(0.0);
        let real_error = ModeAmplitude::new(Complex64::new(1.0e-6, 0.0));
        let imaginary_error = ModeAmplitude::new(Complex64::new(0.0, 1.0e-6));

        let from_real = normalized_mode_error(real_error, reference, absolute, 1.0e-9);
        let from_imaginary = normalized_mode_error(imaginary_error, reference, absolute, 1.0e-9);

        assert_eq!(from_real, from_imaginary);
        assert!(from_real > 1.0);
    }

    #[test]
    fn a_vanished_mode_falls_back_to_the_absolute_tolerance() {
        let error = ModeAmplitude::new(Complex64::new(1.0e-13, 0.0));
        let absolute = ModeAmplitude::<f64>::new(1.0e-12);

        let ratio = normalized_mode_error(error, ModeAmplitude::<f64>::new(0.0), absolute, 1.0e-9);

        assert!(ratio < 1.0);
    }

    #[test]
    fn a_zero_component_does_not_dominate_the_mode_norm() {
        // An axis-aligned trajectory puts exact zeros off the diagonal of the
        // mode matrix. Normalizing each component by its own modulus would
        // give those an unbounded ratio as soon as rounding made their error
        // non-zero; normalizing by the matrix keeps them harmless.
        let system = system();
        let background = MultifieldState::new(
            [Field::new(12.0), Field::new(0.0)],
            [FieldVelocity::new(-3.0e-6), FieldVelocity::new(0.0)],
            ScaleFactor::new(1.0),
        );
        let state = system.bunch_davies_state(background, 100.0).unwrap();
        assert_eq!(
            state.modes.rescaled_amplitude[0][1].complex_magnitude(),
            0.0
        );

        let tolerances = ErrorTolerances::new(
            PerturbationTolerance::background_in_planck_units(1.0e-12),
            1.0e-8,
        );
        let mut delta = zero_delta();
        // Rounding-sized noise in the component that sits at zero.
        delta.amplitude[0][1] = ModeAmplitude::new(Complex64::new(1.0e-16, 0.0));

        let ratio = system.error_ratio(&state, &state, &delta, &tolerances);

        assert!(ratio.is_finite(), "ratio was {ratio}");
        assert!(
            ratio < 1.0,
            "a rounding-sized error was treated as large: {ratio}"
        );
    }

    fn zero_delta() -> PerturbationDelta<2> {
        PerturbationDelta {
            background: MultifieldStateDelta {
                phi: [Field::new(0.0); 2],
                pi: [FieldVelocity::new(0.0); 2],
                a: ScaleFactor::new(0.0),
            },
            amplitude: [[ModeAmplitude::new(Complex64::new(0.0, 0.0)); 2]; 2],
            velocity: [[ModeVelocity::new(Complex64::new(0.0, 0.0)); 2]; 2],
        }
    }

    #[test]
    fn a_purely_imaginary_mode_error_still_drives_the_step_decision() {
        let system = system();
        let state = system.bunch_davies_state(background(), 100.0).unwrap();
        let tolerances = ErrorTolerances::new(
            PerturbationTolerance::background_in_planck_units(1.0e-12),
            1.0e-8,
        );

        assert_eq!(
            system.error_ratio(&state, &state, &zero_delta(), &tolerances),
            0.0
        );

        let mut imaginary = zero_delta();
        imaginary.amplitude[0][0] = ModeAmplitude::new(Complex64::new(0.0, 1.0));

        assert!(system.error_ratio(&state, &state, &imaginary, &tolerances) > 1.0);
    }

    #[test]
    fn an_adaptive_step_accepts_a_small_step_and_rejects_a_huge_one() {
        let system = system();
        let state = system.bunch_davies_state(background(), 100.0).unwrap();
        let tolerances = ErrorTolerances::new(
            PerturbationTolerance::background_in_planck_units(1.0e-12),
            1.0e-8,
        );

        let small = adaptive_step(&system, Time::new(0.0), &state, Time::new(1.0), &tolerances);
        let huge = adaptive_step(
            &system,
            Time::new(0.0),
            &state,
            Time::new(1.0e6),
            &tolerances,
        );

        assert!(small.accepted);
        assert!(!huge.accepted);
        assert!(huge.error_ratio > small.error_ratio);
    }

    #[test]
    fn bunch_davies_modes_have_identity_wronskian() {
        let system = system();
        let state = system.bunch_davies_state(background(), 100.0).unwrap();
        let wronskian = system.wronskian(&state);

        for (row, entries) in wronskian.iter().enumerate() {
            for (column, value) in entries.iter().enumerate() {
                let expected = f64::from(row == column);
                assert!((value.re - expected).abs() < 1.0e-12);
                assert!(value.im.abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn initialization_rejects_a_mode_too_close_to_the_horizon() {
        let system = PerturbationSystem::new(system().background, Mass::new(1.0e-8));

        assert!(matches!(
            system.bunch_davies_state(background(), 100.0),
            Err(ModeInitializationError::NotDeepInsideHorizon { .. })
        ));
    }

    #[test]
    fn generic_integrator_advances_background_and_complex_modes() {
        let system = system();
        let initial = system.bunch_davies_state(background(), 100.0).unwrap();

        let advanced = rk4_step(&system, Time::new(0.0), &initial, Time::new(1.0));

        assert!(advanced.background.a.into_value() > initial.background.a.into_value());
        assert_ne!(
            advanced.modes.rescaled_amplitude,
            initial.modes.rescaled_amplitude
        );
    }

    #[test]
    fn wronskian_is_conserved_by_a_small_rk4_step() {
        let system = system();
        let initial = system.bunch_davies_state(background(), 100.0).unwrap();

        let advanced = rk4_step(&system, Time::new(0.0), &initial, Time::new(1.0e-3));
        let wronskian = system.wronskian(&advanced);

        for (row, entries) in wronskian.iter().enumerate() {
            for (column, value) in entries.iter().enumerate() {
                let expected = f64::from(row == column);
                assert!((value.re - expected).abs() < 1.0e-12);
                assert!(value.im.abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn conformal_state_round_trips_the_background_and_wronskian() {
        let cosmic_system = system();
        let conformal_system =
            ConformalPerturbationSystem::new(cosmic_system.background, cosmic_system.k);
        let cosmic = cosmic_system
            .bunch_davies_state(background(), 100.0)
            .unwrap();
        let conformal = conformal_system
            .bunch_davies_state(cosmic.background, 100.0)
            .unwrap();

        assert_eq!(conformal.cosmic(), cosmic);
        assert_eq!(
            conformal_system.wronskian(&conformal),
            cosmic_system.wronskian(&cosmic)
        );
    }

    #[test]
    fn conformal_derivative_is_the_coordinate_transform_of_cosmic_time() {
        let cosmic_system = system();
        let conformal_system =
            ConformalPerturbationSystem::new(cosmic_system.background, cosmic_system.k);
        let cosmic_state = cosmic_system
            .bunch_davies_state(background(), 100.0)
            .unwrap();
        let conformal_state = conformal_system
            .bunch_davies_state(cosmic_state.background, 100.0)
            .unwrap();
        let cosmic = cosmic_system.derivative(Time::new(0.0), &cosmic_state);
        let conformal = conformal_system.derivative(ConformalTime::new(0.0), &conformal_state);
        let a = cosmic_state.background.a.into_value();
        let hubble = cosmic_system.background.hubble(&cosmic_state.background);

        for field in 0..2 {
            assert_close(
                conformal.phi[field].into_value(),
                cosmic.background.dphi_dt[field].into_value() * a,
            );
            assert_close(
                conformal.pi[field].into_value(),
                cosmic.background.dpi_dt[field].into_value() * a,
            );
            for mode in 0..2 {
                assert_complex_close(
                    conformal.amplitude[field][mode].into_value(),
                    cosmic.amplitude[field][mode].into_value() * a,
                );
                // q = a p, hence dq/deta = a^2 (dp/dt + H p).
                assert_complex_close(
                    conformal.velocity[field][mode].into_value(),
                    (cosmic.velocity[field][mode]
                        + cosmic_state.modes.rescaled_velocity[field][mode] * hubble)
                        .into_value()
                        * (a * a),
                );
            }
        }
        assert_close(
            conformal.log_scale_factor.into_value(),
            a * hubble.into_value(),
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0e-30);
        assert!(
            (actual - expected).abs() / scale < 1.0e-12,
            "actual={actual:e}, expected={expected:e}"
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        let scale = actual.norm().max(expected.norm()).max(1.0e-30);
        assert!(
            (actual - expected).norm() / scale < 1.0e-12,
            "actual={actual:e}, expected={expected:e}"
        );
    }
}
