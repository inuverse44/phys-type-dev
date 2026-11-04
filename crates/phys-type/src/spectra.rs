//! Power spectra built from a mode matrix: the invariant spectra for any
//! number of fields, and the adiabatic and entropy projection for two fields.
//!
//! The mode covariance is complex and carries two upper field-space indices.
//! Projecting it onto the adiabatic and entropy directions requires lowered
//! basis covectors, so an accidental same-variance contraction does not
//! compile. A projected spectrum becomes a real number only after its
//! imaginary residual has been checked against an explicit tolerance.

use core::f64::consts::PI;

use num_complex::Complex64;

use crate::{
    ComplexMagnitude, ContravariantField, CovariantField, CurvedFieldSpaceGeometry, Dimensionless,
    FieldMetric, FieldTensor2, Hubble, KinematicBasis2, KinematicBasisError, M0, M2, MMinus1,
    MMinusHalf, Mass, ModeAmplitude, Modes, PerturbationState, PerturbationSystem, Quantity,
    ScaleFactor, TwiceDifferentiableMultifieldPotential, Upper,
};

/// Mode covariance `Sigma^{IJ} = sum_A U^I_A U^{J*}_A`.
pub type ModeCovariance<const N: usize> = FieldTensor2<Upper, Upper, MMinus1, Complex64, N>;

/// Field power spectrum matrix `P_Q^{IJ} = k^3 Sigma^{IJ} / (2 pi^2)`.
pub type FieldPowerSpectrum<const N: usize> = FieldTensor2<Upper, Upper, M2, Complex64, N>;

/// A dimensionless spectrum of the curvature or entropy perturbation.
pub type DimensionlessSpectrum = Dimensionless;

/// Tolerances for the reality and positivity checks that guard a projection.
///
/// Every tolerance is relative to the largest modulus in the spectrum matrix,
/// so the same settings apply at any overall normalization.
///
/// The three are not equally tight, because they measure different things.
/// Hermiticity holds by construction and fails only through rounding. The
/// imaginary residual instead measures accumulated integration error: the
/// antisymmetric part of `Sigma` is the equal-time field commutator, which
/// vanishes at Bunch-Davies initialization and is conserved by the mode
/// equations, so whatever survives to the output time was produced by the
/// integrator. Choose it together with the integration tolerances rather than
/// in isolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrumTolerances {
    /// Largest accepted relative violation of `P^{IJ} = P^{JI*}`.
    pub hermiticity: f64,
    /// Largest accepted relative imaginary part of a projected spectrum.
    pub imaginary_residual: f64,
    /// Largest accepted overshoot of `|r_C|` beyond one.
    pub correlation_overshoot: f64,
}

impl Default for SpectrumTolerances {
    fn default() -> Self {
        Self {
            hermiticity: 1.0e-8,
            imaginary_residual: 1.0e-6,
            correlation_overshoot: 1.0e-8,
        }
    }
}

/// Why a mode covariance could not be turned into physical spectra.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectrumError {
    /// The spectrum matrix failed the Hermiticity check.
    NotHermitian {
        /// The observed relative violation.
        residual: f64,
        /// The tolerance it exceeded.
        tolerance: f64,
    },
    /// A projected spectrum kept an imaginary part too large to discard.
    ImaginaryResidual {
        /// The observed relative imaginary part.
        residual: f64,
        /// The tolerance it exceeded.
        tolerance: f64,
    },
    /// An autocorrelation spectrum was not positive, so no correlation
    /// coefficient is defined.
    NonPositiveSpectrum {
        /// `P_R`, which must be positive.
        curvature: f64,
        /// `P_S`, which must be positive.
        entropy: f64,
    },
    /// The correlation coefficient left `[-1, 1]` by more than the tolerance.
    CorrelationOutOfRange {
        /// The unclamped coefficient.
        value: f64,
        /// The tolerance it exceeded.
        tolerance: f64,
    },
    /// The adiabatic and entropy directions could not be constructed.
    KinematicBasis(KinematicBasisError),
}

/// Spectra projected onto the adiabatic and entropy directions.
///
/// The field spectra keep the dimension `M^2`. The curvature and entropy
/// spectra are dimensionless because each carries two factors of
/// `H / dot_sigma`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedSpectra {
    /// `P_{Q_sigma}`, the adiabatic field spectrum.
    pub field_adiabatic: Quantity<M2, f64>,
    /// `P_{Q_s}`, the entropy field spectrum.
    pub field_entropy: Quantity<M2, f64>,
    /// `C_{Q_sigma Q_s}`, the signed field cross spectrum.
    pub field_cross: Quantity<M2, f64>,
    /// `P_R`, the curvature perturbation spectrum.
    pub curvature: DimensionlessSpectrum,
    /// `P_S`, the entropy perturbation spectrum.
    pub entropy: DimensionlessSpectrum,
    /// `C_RS`, the signed curvature-entropy cross spectrum.
    pub cross: DimensionlessSpectrum,
    /// `r_C = C_RS / sqrt(P_R P_S)`, clamped to `[-1, 1]`.
    pub correlation: f64,
}

/// Builds `Sigma^{IJ} = sum_A U^I_A U^{J*}_A` from a perturbation state.
///
/// The modes are integrated as `v^I_A = a U^I_A`, so this is the one place
/// where the rescaling is undone: the sum carries `1 / a^2`. It runs over the
/// independent vacuum oscillators, so the result is Hermitian by construction
/// up to floating-point rounding. Taking the complete state prevents modes
/// from accidentally being paired with a scale factor from another time.
pub fn mode_covariance<const N: usize>(state: &PerturbationState<N>) -> ModeCovariance<N> {
    mode_covariance_from_modes(&state.modes, state.background.a)
}

fn mode_covariance_from_modes<const N: usize>(
    modes: &Modes<N>,
    scale_factor: ScaleFactor,
) -> ModeCovariance<N> {
    let squared = scale_factor * scale_factor;

    ModeCovariance::new(core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            let rescaled = (0..N).fold(
                Quantity::<MMinus1, Complex64>::new(Complex64::new(0.0, 0.0)),
                |sum, mode| {
                    let conjugate =
                        ModeAmplitude::new(modes.rescaled_amplitude[column][mode].value().conj());

                    sum + modes.rescaled_amplitude[row][mode] * conjugate
                },
            );

            rescaled / squared
        })
    }))
}

/// Applies the `k^3 / (2 pi^2)` normalization to a mode covariance.
pub fn field_power_spectrum<const N: usize>(
    k: Mass,
    covariance: &ModeCovariance<N>,
) -> FieldPowerSpectrum<N> {
    let normalization = k * k * k / (2.0 * PI * PI);

    FieldPowerSpectrum::new(
        covariance
            .components()
            .map(|row| row.map(|entry| normalization * entry)),
    )
}

/// The largest modulus in a spectrum matrix, used as the scale for every
/// relative tolerance.
pub(crate) fn spectrum_scale<const N: usize>(spectrum: &FieldPowerSpectrum<N>) -> f64 {
    spectrum
        .components()
        .iter()
        .flatten()
        .map(ComplexMagnitude::complex_magnitude)
        .fold(0.0, f64::max)
}

/// The relative violation of `P^{IJ} = P^{JI*}`.
///
/// A matrix whose entries all vanish is Hermitian, and reports zero rather
/// than dividing by an empty scale.
pub fn hermiticity_residual<const N: usize>(spectrum: &FieldPowerSpectrum<N>) -> f64 {
    let scale = spectrum_scale(spectrum);
    if scale == 0.0 {
        return 0.0;
    }

    let components = spectrum.components();
    let mut residual: f64 = 0.0;
    for (row, entries) in components.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            let difference = entry.value() - components[column][row].value().conj();
            residual = residual.max(difference.norm() / scale);
        }
    }

    residual
}

/// Contracts a spectrum matrix with two lowered basis covectors.
///
/// Both indices of `P^{IJ}` are upper, so only covectors can be supplied.
fn project_complex(
    spectrum: &FieldPowerSpectrum<2>,
    left: &CovariantField<M0, f64, 2>,
    right: &CovariantField<M0, f64, 2>,
) -> Quantity<M2, Complex64> {
    let components = spectrum.components();
    let left = left.components();
    let right = right.components();

    (0..2)
        .flat_map(|row| (0..2).map(move |column| (row, column)))
        .fold(
            Quantity::<M2, Complex64>::new(Complex64::new(0.0, 0.0)),
            |sum, (row, column)| {
                let weight = left[row].value() * right[column].value();

                sum + components[row][column] * weight
            },
        )
}

/// Discards the imaginary part of a projected spectrum after checking that it
/// is negligible relative to `scale`.
pub(crate) fn discard_imaginary(
    value: Quantity<M2, Complex64>,
    scale: f64,
    tolerance: f64,
) -> Result<Quantity<M2, f64>, SpectrumError> {
    let value = value.into_value();
    if scale > 0.0 {
        let residual = value.im.abs() / scale;
        if residual > tolerance {
            return Err(SpectrumError::ImaginaryResidual {
                residual,
                tolerance,
            });
        }
    }

    Ok(Quantity::new(value.re))
}

/// The dimensionless curvature and entropy amplitude of each independent
/// mode, as `[[R_1, R_2], [S_1, S_2]]`.
///
/// Summing `|R_A|^2` over the independent modes reproduces `P_R`, and
/// likewise for the entropy row, so this is the same content as
/// [`ProjectedSpectra`] before the sum over vacuum oscillators discards which
/// solution each contribution came from. The super-horizon transfer matrix
/// needs that information, because it relates two independent solutions
/// rather than their sum.
pub type ModeProjections = [[Complex64; 2]; 2];

/// Projects each independent mode onto the curvature and entropy directions.
fn project_modes_in_basis(
    modes: &Modes<2>,
    scale_factor: ScaleFactor,
    metric: &FieldMetric<2>,
    basis: &KinematicBasis2,
    hubble: Hubble,
    k: Mass,
) -> ModeProjections {
    let adiabatic = metric.lower(&basis.tangent);
    let entropy = metric.lower(&basis.normal);
    let directions = [adiabatic, entropy];
    // R = (H / dot_sigma) Q_sigma, and P_R = (k^3 / 2 pi^2) |R|^2 summed over
    // modes, so each amplitude carries k^(3/2) / (pi sqrt 2).
    let normalization = (k * k * k).sqrt() / (PI * 2.0_f64.sqrt());
    let transfer = hubble / basis.speed;

    core::array::from_fn(|direction| {
        core::array::from_fn(|mode| {
            let projected = (0..2).fold(
                Quantity::<MMinusHalf, Complex64>::new(Complex64::new(0.0, 0.0)),
                |sum, field| {
                    let weight = *directions[direction].components()[field].value();
                    let amplitude = modes.rescaled_amplitude[field][mode] / scale_factor;

                    sum + amplitude * weight
                },
            );

            (normalization * transfer * projected).into_value()
        })
    })
}

/// Projects a spectrum matrix onto an adiabatic and entropy basis.
///
/// `metric` lowers the basis vectors, and `hubble` together with the basis
/// speed `dot_sigma` supplies the `H / dot_sigma` normalization that makes the
/// curvature and entropy spectra dimensionless.
pub fn project(
    spectrum: &FieldPowerSpectrum<2>,
    metric: &FieldMetric<2>,
    basis: &KinematicBasis2,
    hubble: Hubble,
    tolerances: SpectrumTolerances,
) -> Result<ProjectedSpectra, SpectrumError> {
    let residual = hermiticity_residual(spectrum);
    if residual > tolerances.hermiticity {
        return Err(SpectrumError::NotHermitian {
            residual,
            tolerance: tolerances.hermiticity,
        });
    }

    let adiabatic = metric.lower(&basis.tangent);
    let entropy = metric.lower(&basis.normal);
    let scale = spectrum_scale(spectrum);

    let field_adiabatic = discard_imaginary(
        project_complex(spectrum, &adiabatic, &adiabatic),
        scale,
        tolerances.imaginary_residual,
    )?;
    let field_entropy = discard_imaginary(
        project_complex(spectrum, &entropy, &entropy),
        scale,
        tolerances.imaginary_residual,
    )?;
    let field_cross = discard_imaginary(
        project_complex(spectrum, &adiabatic, &entropy),
        scale,
        tolerances.imaginary_residual,
    )?;

    // R = (H / dot_sigma) Q_sigma, so each spectrum picks up two factors.
    let transfer = hubble / basis.speed;
    let normalization = transfer * transfer;
    let curvature = normalization * field_adiabatic;
    let entropy_spectrum = normalization * field_entropy;
    let cross = normalization * field_cross;

    let correlation = correlation_coefficient(
        curvature,
        entropy_spectrum,
        cross,
        tolerances.correlation_overshoot,
    )?;

    Ok(ProjectedSpectra {
        field_adiabatic,
        field_entropy,
        field_cross,
        curvature,
        entropy: entropy_spectrum,
        cross,
        correlation,
    })
}

/// Forms `r_C = C_RS / sqrt(P_R P_S)` and clamps a tolerable overshoot.
fn correlation_coefficient(
    curvature: DimensionlessSpectrum,
    entropy: DimensionlessSpectrum,
    cross: DimensionlessSpectrum,
    tolerance: f64,
) -> Result<f64, SpectrumError> {
    let curvature = curvature.into_value();
    let entropy = entropy.into_value();
    if !(curvature > 0.0 && entropy > 0.0) {
        return Err(SpectrumError::NonPositiveSpectrum { curvature, entropy });
    }

    let value = cross.into_value() / (curvature * entropy).sqrt();
    if value.abs() > 1.0 + tolerance {
        return Err(SpectrumError::CorrelationOutOfRange { value, tolerance });
    }

    Ok(value.clamp(-1.0, 1.0))
}

impl<P, G> PerturbationSystem<P, G, 2>
where
    P: TwiceDifferentiableMultifieldPotential<2>,
    G: CurvedFieldSpaceGeometry<2>,
{
    /// Projects every independent solution onto the instantaneous curvature
    /// and entropy directions.
    ///
    /// The system and complete state supply `k`, `a`, the metric, the basis,
    /// and `H` together, so values belonging to different evolution times
    /// cannot be combined accidentally.
    pub fn mode_projections(
        &self,
        state: &PerturbationState<2>,
        previous_normal: Option<&ContravariantField<M0, f64, 2>>,
    ) -> Result<ModeProjections, SpectrumError> {
        let kinematics = self
            .background
            .oriented_kinematics(&state.background, previous_normal)
            .map_err(SpectrumError::KinematicBasis)?;
        let metric = self.background.geometry.metric(&state.background.phi);
        let hubble = self.background.hubble(&state.background);

        Ok(project_modes_in_basis(
            &state.modes,
            state.background.a,
            &metric,
            &kinematics.basis,
            hubble,
            self.k,
        ))
    }

    /// Projects the spectra of one evolved state onto its instantaneous
    /// adiabatic and entropy directions.
    ///
    /// `previous_normal` carries the entropy orientation forward between
    /// output times, exactly as for the background kinematics.
    pub fn spectra(
        &self,
        state: &PerturbationState<2>,
        previous_normal: Option<&ContravariantField<M0, f64, 2>>,
        tolerances: SpectrumTolerances,
    ) -> Result<ProjectedSpectra, SpectrumError> {
        let kinematics = self
            .background
            .oriented_kinematics(&state.background, previous_normal)
            .map_err(SpectrumError::KinematicBasis)?;
        let metric = self.background.geometry.metric(&state.background.phi);
        let hubble = self.background.hubble(&state.background);
        let spectrum = field_power_spectrum(self.k, &mode_covariance(state));

        project(&spectrum, &metric, &kinematics.basis, hubble, tolerances)
    }
}

/// Spectra of any number of fields, defined without an entropy basis.
///
/// With the field spectrum `P^{IJ}`, the tangent `e_sigma`, the entropy
/// projector `perp^{IJ}`, and `H / dot_sigma`, the curvature spectrum is the
/// adiabatic projection and `S^{IJ} = (H/dot_sigma)^2 perp^I_K perp^J_L P^{KL}`
/// is the entropy covariance. Its trace, its eigenvalues (the principal
/// entropy spectra, along axes set by the perturbations), and the norm of
/// the cross vector `C^I = (H/dot_sigma)^2 perp^I_K e_sigma_L Re P^{KL}` are
/// invariant under changes of field coordinates; see
/// `docs/n-field-perturbations.md`. For two fields they equal the oriented
/// [`ProjectedSpectra`] up to the sign of the cross spectrum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvariantSpectra<const N: usize> {
    /// `P_R`.
    pub curvature: DimensionlessSpectrum,
    /// `P_S = S^I_I`, the total entropy power.
    pub entropy: DimensionlessSpectrum,
    /// Eigenvalues of `S^I_J` in decreasing order. The last one belongs to
    /// the adiabatic direction and vanishes up to round-off; the first
    /// `N - 1` are the principal entropy spectra.
    pub principal_entropy: [f64; N],
    /// `|C_RS| = sqrt(G_IJ C^I C^J)`.
    pub cross_norm: DimensionlessSpectrum,
    /// `r_C = |C_RS| / sqrt(P_R P_S)`; `None` for one field, where there is
    /// no entropy.
    pub correlation: Option<f64>,
}

/// Eigenvalues of a real symmetric matrix by cyclic Jacobi rotations, in
/// decreasing order.
fn symmetric_eigenvalues<const N: usize>(mut matrix: [[f64; N]; N]) -> [f64; N] {
    for _ in 0..100 {
        let off: f64 = (0..N)
            .flat_map(|i| (0..N).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| matrix[i][j] * matrix[i][j])
            .sum();
        let scale: f64 = (0..N).map(|i| matrix[i][i] * matrix[i][i]).sum::<f64>() + off;
        if off <= 1.0e-30 * scale {
            break;
        }
        for p in 0..N {
            for q in (p + 1)..N {
                if matrix[p][q] == 0.0 {
                    continue;
                }
                let theta = (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in matrix.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
                let (row_p, row_q) = (matrix[p], matrix[q]);
                matrix[p] = core::array::from_fn(|k| c * row_p[k] - s * row_q[k]);
                matrix[q] = core::array::from_fn(|k| s * row_p[k] + c * row_q[k]);
            }
        }
    }
    let mut values: [f64; N] = core::array::from_fn(|i| matrix[i][i]);
    values.sort_by(|a, b| b.total_cmp(a));
    values
}

/// Projects a field spectrum of `N` fields onto basis-free curvature and
/// entropy spectra.
pub fn invariant_spectra<const N: usize>(
    spectrum: &FieldPowerSpectrum<N>,
    metric: &FieldMetric<N>,
    kinematics: &crate::Kinematics<N>,
    hubble: Hubble,
    tolerances: SpectrumTolerances,
) -> Result<InvariantSpectra<N>, SpectrumError> {
    let residual = hermiticity_residual(spectrum);
    if residual > tolerances.hermiticity {
        return Err(SpectrumError::NotHermitian {
            residual,
            tolerance: tolerances.hermiticity,
        });
    }
    let g = |i: usize, j: usize| metric.components()[i][j].into_value();
    let p = |i: usize, j: usize| spectrum.components()[i][j].into_value().re;
    let e: [f64; N] = kinematics.tangent.components().map(Quantity::into_value);
    let transfer = (hubble / kinematics.speed).into_value();
    let weight = transfer * transfer;
    // Lowered tangent and the mixed projector perp^I_J = delta^I_J - e^I e_J.
    let e_lower: [f64; N] = core::array::from_fn(|i| (0..N).map(|j| g(i, j) * e[j]).sum());
    let perp = |i: usize, j: usize| f64::from(i == j) - e[i] * e_lower[j];
    let curvature: f64 = weight
        * (0..N)
            .flat_map(|i| (0..N).map(move |j| (i, j)))
            .map(|(i, j)| e_lower[i] * e_lower[j] * p(i, j))
            .sum::<f64>();
    let entropy_upper: [[f64; N]; N] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            weight
                * (0..N)
                    .flat_map(|k| (0..N).map(move |l| (k, l)))
                    .map(|(k, l)| perp(i, k) * perp(j, l) * p(k, l))
                    .sum::<f64>()
        })
    });
    let cross: [f64; N] = core::array::from_fn(|i| {
        weight
            * (0..N)
                .flat_map(|k| (0..N).map(move |l| (k, l)))
                .map(|(k, l)| perp(i, k) * e_lower[l] * p(k, l))
                .sum::<f64>()
    });
    // S^I_I = S^{IJ} G_JI.
    let entropy: f64 = (0..N)
        .flat_map(|i| (0..N).map(move |j| (i, j)))
        .map(|(i, j)| entropy_upper[i][j] * g(j, i))
        .sum();
    let cross_norm = (0..N)
        .flat_map(|i| (0..N).map(move |j| (i, j)))
        .map(|(i, j)| g(i, j) * cross[i] * cross[j])
        .sum::<f64>()
        .max(0.0)
        .sqrt();
    // Eigenvalues of S^I_J equal those of L^T S L for G = L L^T.
    let mut cholesky = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| cholesky[i][k] * cholesky[j][k]).sum();
            if i == j {
                cholesky[i][i] = (g(i, i) - sum).max(0.0).sqrt();
            } else {
                cholesky[i][j] = (g(i, j) - sum) / cholesky[j][j];
            }
        }
    }
    let symmetric: [[f64; N]; N] = core::array::from_fn(|a| {
        core::array::from_fn(|b| {
            (0..N)
                .flat_map(|i| (0..N).map(move |j| (i, j)))
                .map(|(i, j)| cholesky[i][a] * entropy_upper[i][j] * cholesky[j][b])
                .sum()
        })
    });
    let principal_entropy = symmetric_eigenvalues(symmetric);
    let correlation = if N == 1 {
        None
    } else {
        if !(curvature > 0.0 && entropy > 0.0) {
            return Err(SpectrumError::NonPositiveSpectrum { curvature, entropy });
        }
        let value = cross_norm / (curvature * entropy).sqrt();
        if value > 1.0 + tolerances.correlation_overshoot {
            return Err(SpectrumError::CorrelationOutOfRange {
                value,
                tolerance: tolerances.correlation_overshoot,
            });
        }
        Some(value.min(1.0))
    };
    Ok(InvariantSpectra {
        curvature: Quantity::new(curvature),
        entropy: Quantity::new(entropy),
        principal_entropy,
        cross_norm: Quantity::new(cross_norm),
        correlation,
    })
}

impl<P, G, const N: usize> PerturbationSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    /// Projects one evolved state onto basis-free curvature and entropy
    /// spectra, for any number of fields.
    pub fn invariant_spectra(
        &self,
        state: &PerturbationState<N>,
        tolerances: SpectrumTolerances,
    ) -> Result<InvariantSpectra<N>, SpectrumError> {
        let kinematics = self
            .background
            .kinematics(&state.background)
            .map_err(SpectrumError::KinematicBasis)?;
        let metric = self.background.geometry.metric(&state.background.phi);
        let hubble = self.background.hubble(&state.background);
        let spectrum = field_power_spectrum(self.k, &mode_covariance(state));
        invariant_spectra(&spectrum, &metric, &kinematics, hubble, tolerances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CovariantMultifieldSystem, Field, FieldVelocity, FlatFieldSpace, ModeVelocity,
        MultifieldState, QuadraticMultifieldPotential, ScaleFactor, Time, rk4_step,
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

    fn spectrum_from(entries: [[Complex64; 2]; 2]) -> FieldPowerSpectrum<2> {
        FieldPowerSpectrum::new(entries.map(|row| row.map(Quantity::new)))
    }

    fn euclidean_basis(speed: FieldVelocity) -> KinematicBasis2 {
        KinematicBasis2 {
            tangent: ContravariantField::new([Quantity::new(1.0), Quantity::new(0.0)]),
            normal: ContravariantField::new([Quantity::new(0.0), Quantity::new(1.0)]),
            speed,
        }
    }

    #[test]
    fn mode_covariance_is_hermitian() {
        let system = system();
        let state = system.bunch_davies_state(background(), 100.0).unwrap();
        let spectrum = field_power_spectrum(system.k, &mode_covariance(&state));

        assert!(hermiticity_residual(&spectrum) < 1.0e-14);
    }

    #[test]
    fn bunch_davies_spectrum_matches_the_sub_horizon_vacuum() {
        let system = system();
        let initial = background();
        let state = system.bunch_davies_state(initial, 100.0).unwrap();
        let spectrum = field_power_spectrum(system.k, &mode_covariance(&state));

        // With U^I_A = e_A^I / (a sqrt(2k)), the covariance is the inverse
        // metric over 2 k a^2, so P^{IJ} = (k/a)^2 G^{IJ} / (4 pi^2).
        let physical_wavenumber = (system.k / state.background.a).into_value();
        let expected = physical_wavenumber.powi(2) / (4.0 * PI * PI);
        let components = spectrum.components();

        for (row, entries) in components.iter().enumerate() {
            for (column, entry) in entries.iter().enumerate() {
                let value = entry.value();
                let diagonal = expected * f64::from(row == column);
                assert!((value.re - diagonal).abs() < 1.0e-12 * expected);
                assert!(value.im.abs() < 1.0e-12 * expected);
            }
        }
    }

    #[test]
    fn projection_normalizes_with_the_squared_transfer_factor() {
        let spectrum = spectrum_from([
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(9.0, 0.0)],
        ]);
        let hubble = Hubble::new(2.0);
        let speed = FieldVelocity::new(8.0);

        let projected = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(speed),
            hubble,
            SpectrumTolerances::default(),
        )
        .unwrap();

        let transfer = (hubble.into_value() / speed.into_value()).powi(2);
        assert!((projected.field_adiabatic.into_value() - 4.0).abs() < 1.0e-12);
        assert!((projected.field_entropy.into_value() - 9.0).abs() < 1.0e-12);
        assert!((projected.field_cross.into_value() - 1.0).abs() < 1.0e-12);
        assert!((projected.curvature.into_value() - 4.0 * transfer).abs() < 1.0e-15);
        assert!((projected.entropy.into_value() - 9.0 * transfer).abs() < 1.0e-15);
        assert!((projected.cross.into_value() - transfer).abs() < 1.0e-15);
        assert!((projected.correlation - 1.0 / 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn a_negative_cross_spectrum_keeps_its_sign() {
        let spectrum = spectrum_from([
            [Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0)],
            [Complex64::new(-1.0, 0.0), Complex64::new(9.0, 0.0)],
        ]);

        let projected = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(FieldVelocity::new(1.0)),
            Hubble::new(1.0),
            SpectrumTolerances::default(),
        )
        .unwrap();

        assert!(projected.field_cross.into_value() < 0.0);
        assert!(projected.cross.into_value() < 0.0);
        assert!((projected.correlation + 1.0 / 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn a_non_hermitian_matrix_is_rejected_before_projection() {
        let spectrum = spectrum_from([
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(2.0, 0.0), Complex64::new(9.0, 0.0)],
        ]);

        let error = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(FieldVelocity::new(1.0)),
            Hubble::new(1.0),
            SpectrumTolerances::default(),
        )
        .unwrap_err();

        assert!(matches!(error, SpectrumError::NotHermitian { .. }));
    }

    #[test]
    fn a_hermitian_imaginary_cross_term_is_reported_rather_than_silently_dropped() {
        // This matrix is Hermitian, so it passes the first check, but the
        // adiabatic-entropy projection keeps a large imaginary part.
        let spectrum = spectrum_from([
            [Complex64::new(4.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(9.0, 0.0)],
        ]);

        let error = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(FieldVelocity::new(1.0)),
            Hubble::new(1.0),
            SpectrumTolerances::default(),
        )
        .unwrap_err();

        assert!(matches!(error, SpectrumError::ImaginaryResidual { .. }));
    }

    #[test]
    fn a_correlation_beyond_one_is_rejected() {
        let spectrum = spectrum_from([
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        let error = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(FieldVelocity::new(1.0)),
            Hubble::new(1.0),
            SpectrumTolerances::default(),
        )
        .unwrap_err();

        assert!(matches!(error, SpectrumError::CorrelationOutOfRange { .. }));
    }

    #[test]
    fn a_perfectly_correlated_spectrum_is_clamped_to_one() {
        let spectrum = spectrum_from([
            [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        let projected = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(FieldVelocity::new(1.0)),
            Hubble::new(1.0),
            SpectrumTolerances::default(),
        )
        .unwrap();

        assert_eq!(projected.correlation, 1.0);
    }

    #[test]
    fn an_empty_spectrum_has_no_correlation_coefficient() {
        let spectrum = spectrum_from([[Complex64::new(0.0, 0.0); 2]; 2]);

        let error = project(
            &spectrum,
            &FieldMetric::euclidean(),
            &euclidean_basis(FieldVelocity::new(1.0)),
            Hubble::new(1.0),
            SpectrumTolerances::default(),
        )
        .unwrap_err();

        assert!(matches!(error, SpectrumError::NonPositiveSpectrum { .. }));
    }

    #[test]
    fn the_imaginary_part_of_the_covariance_starts_at_zero_and_stays_small() {
        // The antisymmetric part of Sigma is the equal-time field commutator.
        // It vanishes exactly in the Bunch-Davies vacuum and is conserved, so
        // any growth measures integration error rather than physics.
        let system = system();
        let state = system.bunch_davies_state(background(), 100.0).unwrap();
        let initial = mode_covariance(&state);

        for row in initial.components() {
            for entry in row {
                assert_eq!(entry.value().im, 0.0);
            }
        }

        let advanced = rk4_step(&system, Time::new(0.0), &state, Time::new(1.0));
        let evolved = mode_covariance(&advanced);
        let scale = evolved
            .components()
            .iter()
            .flatten()
            .map(ComplexMagnitude::complex_magnitude)
            .fold(0.0, f64::max);

        for row in evolved.components() {
            for entry in row {
                assert!(entry.value().im.abs() < 1.0e-12 * scale);
            }
        }
    }

    #[test]
    fn evolved_modes_still_produce_projectable_spectra() {
        let system = system();
        let state = system.bunch_davies_state(background(), 100.0).unwrap();
        let advanced = rk4_step(&system, Time::new(0.0), &state, Time::new(1.0));

        let projected = system
            .spectra(&advanced, None, SpectrumTolerances::default())
            .unwrap();

        assert!(projected.curvature.into_value() > 0.0);
        assert!(projected.entropy.into_value() > 0.0);
        assert!(projected.correlation.abs() <= 1.0);
    }

    #[test]
    fn mode_velocities_do_not_enter_the_covariance() {
        let system = system();
        let mut state = system.bunch_davies_state(background(), 100.0).unwrap();
        let before = mode_covariance(&state);

        state.modes.rescaled_velocity = [[ModeVelocity::new(Complex64::new(7.0, -3.0)); 2]; 2];

        assert_eq!(mode_covariance(&state), before);
    }
    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0e-300)
    }

    /// Evolves a Bunch-Davies state with fixed RK4 steps.
    fn evolve<P, G, const N: usize>(
        system: &PerturbationSystem<P, G, N>,
        background: MultifieldState<N>,
        steps: usize,
    ) -> PerturbationState<N>
    where
        P: TwiceDifferentiableMultifieldPotential<N>,
        G: CurvedFieldSpaceGeometry<N>,
    {
        let mut state = system.bunch_davies_state(background, 100.0).unwrap();
        for _ in 0..steps {
            state = rk4_step(system, Time::new(0.0), &state, Time::new(2.0));
        }
        state
    }

    #[test]
    fn invariant_spectra_reproduce_the_oriented_two_field_projection() {
        let background_system = CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(1.0e-6), Mass::new(7.0e-6)]),
            crate::HyperbolicFieldSpace::new(Field::new(5.0)).unwrap(),
        );
        let system = PerturbationSystem::new(background_system, Mass::new(1.0e-2));
        let state = evolve(&system, background(), 40);
        let oriented = system
            .spectra(&state, None, SpectrumTolerances::default())
            .unwrap();
        let general = system
            .invariant_spectra(&state, SpectrumTolerances::default())
            .unwrap();
        assert!(close(
            general.curvature.into_value(),
            oriented.curvature.into_value(),
            1.0e-12
        ));
        assert!(close(
            general.entropy.into_value(),
            oriented.entropy.into_value(),
            1.0e-12
        ));
        assert!(close(
            general.principal_entropy[0],
            oriented.entropy.into_value(),
            1.0e-10
        ));
        assert!(general.principal_entropy[1].abs() < 1.0e-10 * general.entropy.into_value());
        // Inside the horizon the cross spectrum is tiny, so it is compared on
        // the scale sqrt(P_R P_S) that bounds it.
        let bound = (general.curvature.into_value() * general.entropy.into_value()).sqrt();
        assert!(
            (general.cross_norm.into_value() - oriented.cross.into_value().abs()).abs()
                < 1.0e-12 * bound
        );
        assert!((general.correlation.unwrap() - oriented.correlation.abs()).abs() < 1.0e-12);
    }

    /// `V = sum_i m_i^2 (R phi)_i^2 / 2` for a fixed rotation `R`.
    #[derive(Clone, Copy, Debug)]
    struct RotatedQuadratic {
        masses: [f64; 3],
        rotation: [[f64; 3]; 3],
    }

    impl<V> crate::MultifieldPotentialValue<V, 3> for RotatedQuadratic
    where
        V: From<f64>
            + Clone
            + core::ops::Add<V, Output = V>
            + core::ops::Mul<V, Output = V>
            + core::ops::Mul<f64, Output = V>,
    {
        fn value(&self, fields: [Field<V>; 3]) -> crate::EnergyDensity<V> {
            let phi = fields.map(Quantity::into_value);
            let rotated: [V; 3] = core::array::from_fn(|i| {
                (0..3).fold(V::from(0.0), |sum, j| {
                    sum + phi[j].clone() * self.rotation[i][j]
                })
            });
            crate::EnergyDensity::new((0..3).fold(V::from(0.0), |sum, i| {
                sum + rotated[i].clone()
                    * rotated[i].clone()
                    * (0.5 * self.masses[i] * self.masses[i])
            }))
        }
    }

    #[test]
    fn invariant_spectra_do_not_depend_on_the_field_coordinates() {
        let masses = [1.0e-6, 2.3e-6, 4.1e-6];
        let (a, b) = (0.6_f64, 0.3_f64);
        // A rotation about z by a, then about x by b.
        let rz = [
            [a.cos(), -a.sin(), 0.0],
            [a.sin(), a.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let rx = [
            [1.0, 0.0, 0.0],
            [0.0, b.cos(), -b.sin()],
            [0.0, b.sin(), b.cos()],
        ];
        let rotation: [[f64; 3]; 3] = core::array::from_fn(|i| {
            core::array::from_fn(|j| (0..3).map(|k| rx[i][k] * rz[k][j]).sum())
        });
        let identity: [[f64; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|j| f64::from(i == j)));
        let k = Mass::new(1.0e-2);
        let original = PerturbationSystem::new(
            CovariantMultifieldSystem::in_planck_units(
                RotatedQuadratic {
                    masses,
                    rotation: identity,
                },
                FlatFieldSpace,
            ),
            k,
        );
        let rotated = PerturbationSystem::new(
            CovariantMultifieldSystem::in_planck_units(
                RotatedQuadratic { masses, rotation },
                FlatFieldSpace,
            ),
            k,
        );
        // The same physical state in both coordinates: phi' = R^T phi.
        let phi = [10.0, 7.0, 4.0];
        let velocity = [-3.0e-6, -4.0e-6, -2.0e-6];
        let back = |v: [f64; 3]| -> [f64; 3] {
            core::array::from_fn(|j| (0..3).map(|i| rotation[i][j] * v[i]).sum())
        };
        let state_in = |p: [f64; 3], v: [f64; 3]| {
            MultifieldState::new(
                p.map(Field::new),
                v.map(FieldVelocity::new),
                ScaleFactor::new(1.0),
            )
        };
        let a_state = evolve(&original, state_in(phi, velocity), 40);
        let b_state = evolve(&rotated, state_in(back(phi), back(velocity)), 40);
        let a = original
            .invariant_spectra(&a_state, SpectrumTolerances::default())
            .unwrap();
        let b = rotated
            .invariant_spectra(&b_state, SpectrumTolerances::default())
            .unwrap();
        assert!(close(
            a.curvature.into_value(),
            b.curvature.into_value(),
            1.0e-9
        ));
        assert!(close(
            a.entropy.into_value(),
            b.entropy.into_value(),
            1.0e-9
        ));
        let bound = (a.curvature.into_value() * a.entropy.into_value()).sqrt();
        assert!((a.cross_norm.into_value() - b.cross_norm.into_value()).abs() < 1.0e-9 * bound);
        for i in 0..2 {
            assert!(close(
                a.principal_entropy[i],
                b.principal_entropy[i],
                1.0e-9
            ));
        }
        // The entropy eigenvalues add up to the trace; the last one vanishes.
        let sum: f64 = a.principal_entropy.iter().sum();
        assert!(close(sum, a.entropy.into_value(), 1.0e-10));
        assert!(a.principal_entropy[2].abs() < 1.0e-10 * a.entropy.into_value());
    }

    #[test]
    fn one_field_matches_two_fields_with_a_decoupled_spectator() {
        let k = Mass::new(1.0e-2);
        let single = PerturbationSystem::new(
            CovariantMultifieldSystem::in_planck_units(
                QuadraticMultifieldPotential::new([Mass::new(1.0e-6)]),
                FlatFieldSpace,
            ),
            k,
        );
        let pair = PerturbationSystem::new(
            CovariantMultifieldSystem::in_planck_units(
                QuadraticMultifieldPotential::new([Mass::new(1.0e-6), Mass::new(3.0e-5)]),
                FlatFieldSpace,
            ),
            k,
        );
        let one = evolve(
            &single,
            MultifieldState::new(
                [Field::new(12.0)],
                [FieldVelocity::new(-3.0e-6)],
                ScaleFactor::new(1.0),
            ),
            40,
        );
        let two = evolve(
            &pair,
            MultifieldState::new(
                [Field::new(12.0), Field::new(0.0)],
                [FieldVelocity::new(-3.0e-6), FieldVelocity::new(0.0)],
                ScaleFactor::new(1.0),
            ),
            40,
        );
        let a = single
            .invariant_spectra(&one, SpectrumTolerances::default())
            .unwrap();
        let b = pair
            .invariant_spectra(&two, SpectrumTolerances::default())
            .unwrap();
        assert!(close(
            a.curvature.into_value(),
            b.curvature.into_value(),
            1.0e-13
        ));
        assert_eq!(a.correlation, None);
        assert_eq!(a.entropy.into_value(), 0.0);
        assert!(b.cross_norm.into_value() < 1.0e-12 * b.curvature.into_value());
    }
}
