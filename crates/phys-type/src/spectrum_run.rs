//! Driving spectrum runs and exporting their rows.
//!
//! The background survey ([`CovariantMultifieldSystem::survey`]) and the
//! single-mode spectra ([`CovariantMultifieldSystem::mode_spectrum`]) work for
//! any number of fields. The labeled CSV run ([`SpectrumRun`]), with adiabatic
//! and entropy rows, is for two fields.
//!
//! A [`SpectrumRun`] has three stages. A background survey locates the end of
//! inflation and fixes the pivot `k_pivot = a_* H_*`. Each requested wavenumber
//! is then carried forward to its own `x_init = k/(aH)` crossing, initialized
//! in the Bunch-Davies vacuum there, and evolved together with the background.
//! The spectra are projected at one requested output time.
//!
//! The accepted background survey states form one cubic-Hermite trajectory in
//! elapsed e-folds. Every wavenumber then evolves only its mode matrix against
//! that immutable trajectory. The former coupled background-plus-mode path is
//! retained in tests as a numerical reference.

use std::fmt;
use std::io::{self, Write};

use crate::integrate::{
    AdaptiveSystem, ErrorTolerances, StepSizeLimits, System, adaptive_step, next_step_size,
    rk45_step,
};
use crate::spectra::{
    ProjectedSpectra, SpectrumError, SpectrumTolerances, field_power_spectrum, mode_covariance,
    project,
};
use crate::{
    Complex64, ComplexMagnitude, ConformalBackgroundState, ConformalPerturbationState,
    ConformalPerturbationSystem, ConformalPerturbationTolerance, ContravariantField,
    CovariantMultifieldSystem, CurvedFieldSpaceGeometry, FieldSpaceGeometry, KinematicBasisError,
    M0, Mass, ModeAcceleration, ModeAmplitude, ModeInitializationError, ModeVelocity, Modes,
    MultifieldPotential, MultifieldState, MultifieldStateDelta, Quantity, ScaleFactor, Time,
    TwiceDifferentiableMultifieldPotential, TwoFieldKinematics,
};

#[cfg(test)]
use crate::ConformalTime;

/// The event that ends inflation in the background survey.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EndOfInflation {
    /// Slow roll is violated: `epsilon_H = 1`.
    SlowRollViolation,
    /// A separate sector, such as a waterfall field, ends inflation when the
    /// field component `field` crosses `value` from its initial side.
    ///
    /// The survey reports an error if `epsilon_H` reaches one first, because
    /// inflation would then have ended by slow-roll violation instead.
    FieldValue {
        /// Index of the field coordinate.
        field: usize,
        /// The critical value.
        value: crate::Field,
    },
}

/// Everything the run needs beyond the physical model and its initial state.
#[derive(Clone, Copy, Debug)]
pub struct SpectrumRunSettings {
    /// Step size attempted first in every stage.
    pub initial_dt: Time,
    /// Smallest permitted step size.
    pub min_dt: Time,
    /// Largest permitted step size.
    pub max_dt: Time,
    /// Tolerances for the shared adaptive controller.
    ///
    /// The absolute part carries one tolerance per group of components, each
    /// in that group's own dimension. The background gets a floor in Planck
    /// units; the modes get none, because their norm is a modulus that does
    /// not vanish and any floor there would become their accuracy ceiling.
    pub tolerances: ErrorTolerances<ConformalPerturbationTolerance<2>>,
    /// A safety limit on attempted steps in any one stage.
    pub max_steps: usize,
    /// Give up on inflation ending after this many e-folds.
    pub max_e_folds: f64,
    /// The event that ends inflation; `epsilon_H = 1` by default.
    pub end_of_inflation: EndOfInflation,
    /// The background survey continues this many e-folds past the end of
    /// inflation, so that `zeta` can be followed after it; zero by default.
    /// The end of inflation, and every time measured from it, stays at the
    /// event.
    pub e_folds_after_end: f64,
    /// How many e-folds before its own horizon exit each mode is initialized.
    ///
    /// This is the readable form of the depth `x_init = k/(aH)`, which is its
    /// exponential. The trade-off is steep and regular: one more e-fold costs
    /// about `e` times as much work, because that many more sub-horizon
    /// oscillations must be resolved, and buys about `e^2` in accuracy,
    /// because the Bunch-Davies form is approached as `1/x_init^2`.
    pub e_folds_before_exit: f64,
    /// The pivot exits the horizon this many e-folds before inflation ends.
    pub pivot_e_folds_before_end: f64,
    /// Spectra are reported this many e-folds before inflation ends.
    pub output_e_folds_before_end: f64,
    /// Smallest wavenumber of the logarithmic grid, in units of `k_pivot`.
    pub k_min_over_pivot: f64,
    /// Largest wavenumber of the logarithmic grid, in units of `k_pivot`.
    pub k_max_over_pivot: f64,
    /// Number of grid points, which must be at least one.
    pub k_count: usize,
    /// One background row every this many accepted steps.
    pub background_sample_every: usize,
    /// One mode-evolution row every this many accepted steps.
    pub mode_sample_every: usize,
    /// Reality and range tolerances applied when projecting.
    pub spectrum_tolerances: SpectrumTolerances,
}

impl Default for SpectrumRunSettings {
    fn default() -> Self {
        Self {
            initial_dt: Time::new(100.0),
            min_dt: Time::new(1.0e-6),
            max_dt: Time::new(1.0e5),
            tolerances: ErrorTolerances::new(
                ConformalPerturbationTolerance::background_in_planck_units(1.0e-20),
                1.0e-10,
            ),
            max_steps: 1_000_000,
            max_e_folds: 120.0,
            end_of_inflation: EndOfInflation::SlowRollViolation,
            e_folds_after_end: 0.0,
            e_folds_before_exit: 5.0,
            pivot_e_folds_before_end: 55.0,
            output_e_folds_before_end: 5.0,
            k_min_over_pivot: 0.1,
            k_max_over_pivot: 10.0,
            k_count: 9,
            background_sample_every: 20,
            mode_sample_every: 50,
            spectrum_tolerances: SpectrumTolerances::default(),
        }
    }
}

/// Why a spectrum run could not be completed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectrumRunError {
    /// A setting was outside its permitted range.
    InvalidSettings(&'static str),
    /// The end-of-inflation event did not occur within `max_e_folds`.
    InflationDidNotEnd {
        /// The e-folding number actually reached.
        e_folds: f64,
    },
    /// The background left the domain of the field-space geometry.
    OutsideGeometryDomain {
        /// The e-folding number of the first offending step.
        e_folds: f64,
        /// The failed condition.
        error: crate::GeometryDomainError,
    },
    /// `epsilon_H` reached one before the requested field-value event.
    SlowRollEndedFirst {
        /// The e-folding number at which `epsilon_H = 1`.
        e_folds: f64,
    },
    /// Inflation ended before the requested pivot or output time could exist.
    NotEnoughEFolds {
        /// E-folds between the initial state and the end of inflation.
        available: f64,
        /// E-folds the settings asked for before the end of inflation.
        required: f64,
    },
    /// A stage could not meet its tolerances at the minimum step size.
    MinimumStepReached {
        /// The part of the run whose tolerance could not be met.
        stage: SpectrumRunStage,
    },
    /// A stage exhausted its step budget.
    StepLimitReached {
        /// The part of the run that exhausted its step budget.
        stage: SpectrumRunStage,
    },
    /// A mode had not left the horizon by the requested output time, so the
    /// reported spectrum would still be its vacuum rather than a physical
    /// result.
    OutputBeforeHorizonExit {
        /// The wavenumber in Planck units.
        k: f64,
        /// `k/(aH)` at the output time, which must be below one.
        k_over_ah: f64,
    },
    /// A mode could not be placed in the Bunch-Davies vacuum.
    ModeInitialization {
        /// The wavenumber in Planck units.
        k: f64,
        /// Why initialization failed.
        source: ModeInitializationError,
    },
    /// A projection could not produce physical spectra.
    Spectrum {
        /// The wavenumber in Planck units.
        k: f64,
        /// Why the projection failed.
        source: SpectrumError,
    },
    /// The adiabatic/entropy basis was undefined along the background.
    KinematicBasis(KinematicBasisError),
}

/// A stage of a spectrum run, retained in integration errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectrumRunStage {
    /// Locating the end of inflation and sampling the background.
    BackgroundSurvey,
    /// Locating the requested pivot e-folding time.
    Pivot,
    /// Locating the Bunch-Davies initialization point for one mode.
    ModeStart {
        /// Comoving wavenumber in Planck units.
        k: f64,
    },
    /// Evolving one initialized mode to its `k = aH` crossing.
    HorizonExit {
        /// Comoving wavenumber in Planck units.
        k: f64,
    },
    /// Evolving one initialized mode to the requested output time.
    ModeEvolution {
        /// Comoving wavenumber in Planck units.
        k: f64,
    },
}

impl fmt::Display for SpectrumRunStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackgroundSurvey => formatter.write_str("background survey"),
            Self::Pivot => formatter.write_str("pivot search"),
            Self::ModeStart { k } => write!(formatter, "mode-start search for k={k:e} M_pl"),
            Self::HorizonExit { k } => write!(formatter, "horizon-exit search for k={k:e} M_pl"),
            Self::ModeEvolution { k } => write!(formatter, "mode evolution for k={k:e} M_pl"),
        }
    }
}

impl fmt::Display for SpectrumRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSettings(message) => {
                write!(formatter, "invalid spectrum settings: {message}")
            }
            Self::InflationDidNotEnd { e_folds } => {
                write!(
                    formatter,
                    "inflation did not end within {e_folds:e} e-folds"
                )
            }
            Self::OutsideGeometryDomain { e_folds, error } => write!(
                formatter,
                "the background left the geometry's domain after {e_folds:e} e-folds: {error}"
            ),
            Self::SlowRollEndedFirst { e_folds } => write!(
                formatter,
                "epsilon_H reached one after {e_folds:e} e-folds, before the field-value end of inflation"
            ),
            Self::NotEnoughEFolds {
                available,
                required,
            } => write!(
                formatter,
                "only {available:e} e-folds are available, but {required:e} are required"
            ),
            Self::MinimumStepReached { stage } => {
                write!(formatter, "minimum step size reached during {stage}")
            }
            Self::StepLimitReached { stage } => {
                write!(formatter, "step limit reached during {stage}")
            }
            Self::OutputBeforeHorizonExit { k, k_over_ah } => write!(
                formatter,
                "mode k={k:e} M_pl is still inside the horizon at output (k/(aH)={k_over_ah:e})"
            ),
            Self::ModeInitialization { k, source } => {
                write!(
                    formatter,
                    "could not initialize mode k={k:e} M_pl: {source:?}"
                )
            }
            Self::Spectrum { k, source } => {
                write!(formatter, "could not project mode k={k:e} M_pl: {source:?}")
            }
            Self::KinematicBasis(source) => {
                write!(
                    formatter,
                    "could not construct the kinematic basis: {source:?}"
                )
            }
        }
    }
}

impl std::error::Error for SpectrumRunError {}

/// The pivot convention actually realized by a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pivot {
    /// `k_pivot = a_* H_*`.
    pub k: Mass,
    /// `H_*`, the Hubble rate at the pivot crossing.
    pub hubble: crate::Hubble,
    /// `epsilon_H` at the pivot crossing.
    pub epsilon_h: f64,
    /// E-folds from the initial state to the end of inflation.
    pub e_folds_at_end: f64,
    /// E-folds from the initial state to the pivot crossing.
    pub e_folds_at_pivot: f64,
}

/// The result of [`CovariantMultifieldSystem::survey`].
#[derive(Clone, Debug)]
pub struct BackgroundSurvey<const N: usize = 2> {
    /// Background samples: the initial state, every
    /// `background_sample_every`-th accepted step, and the end.
    pub samples: Vec<BackgroundSample<N>>,
    /// The interpolated trajectory.
    pub trajectory: BackgroundTrajectory<N>,
    /// Elapsed e-folds at the end of inflation.
    pub e_folds_at_end: f64,
}

/// One background sample for any number of fields, in Planck units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundSample<const N: usize> {
    /// Elapsed e-folds.
    pub e_folds: f64,
    /// Field values.
    pub phi: [f64; N],
    /// Cosmic-time field velocities.
    pub pi: [f64; N],
    /// Hubble rate.
    pub hubble: f64,
    /// `epsilon_H`.
    pub epsilon_h: f64,
    /// `|eta_perp| = |D_t e_sigma| / H`; zero for one field.
    pub eta_perp: f64,
}

/// One background row, in Planck units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundRecord {
    /// `N = ln(a / a_initial)`.
    pub e_folds: f64,
    /// `phi^I / M_pl`.
    pub phi: [f64; 2],
    /// `(dphi^I/dt) / M_pl^2`.
    pub pi: [f64; 2],
    /// `H / M_pl`.
    pub hubble: f64,
    /// `epsilon_H`.
    pub epsilon_h: f64,
    /// Adiabatic unit vector components.
    pub e_sigma: [f64; 2],
    /// Entropy unit vector components.
    pub e_s: [f64; 2],
    /// Dimensionless signed turn rate.
    pub eta_perp: f64,
}

impl BackgroundRecord {
    /// The CSV header matching [`BackgroundRecord::to_csv_row`].
    pub const CSV_HEADER: &'static str =
        "N,phi_1,phi_2,pi_1,pi_2,H,epsilon_h,e_sigma_1,e_sigma_2,e_s_1,e_s_2,eta_perp";

    /// Formats the row as comma-separated values.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            self.e_folds,
            self.phi[0],
            self.phi[1],
            self.pi[0],
            self.pi[1],
            self.hubble,
            self.epsilon_h,
            self.e_sigma[0],
            self.e_sigma[1],
            self.e_s[0],
            self.e_s[1],
            self.eta_perp
        )
    }
}

/// One spectrum row, in Planck units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrumRecord {
    /// `k / k_pivot`.
    pub k_over_pivot: f64,
    /// Internal `k / M_pl`, which depends on the arbitrary normalization of
    /// the scale factor. Use [`Self::k_over_pivot`] for comparisons.
    pub k_internal_over_m_pl: f64,
    /// `N = ln(a / a_initial)` at this evaluation epoch.
    pub e_folds: f64,
    /// `k / (a H)` at this evaluation epoch.
    pub k_over_ah: f64,
    /// `P^{11} / M_pl^2`.
    pub p_11: f64,
    /// `P^{22} / M_pl^2`.
    pub p_22: f64,
    /// `Re P^{12} / M_pl^2`.
    pub re_p_12: f64,
    /// `Im P^{12} / M_pl^2`.
    ///
    /// Emitted even though it is expected to vanish: twice this imaginary
    /// part forms the antisymmetric equal-time correlator proportional to the
    /// field commutator. It is zero in the vacuum and conserved, so what
    /// survives here measures accumulated integration error.
    pub im_p_12: f64,
    /// `P_{Q_sigma} / M_pl^2`.
    pub p_q_sigma: f64,
    /// `P_{Q_s} / M_pl^2`.
    pub p_q_s: f64,
    /// `C_{Q_sigma Q_s} / M_pl^2`.
    pub c_q_sigma_s: f64,
    /// `P_R`.
    pub p_r: f64,
    /// `P_S`.
    pub p_s: f64,
    /// Signed `C_RS`.
    pub c_rs: f64,
    /// `r_C`.
    pub r_c: f64,
}

impl SpectrumRecord {
    /// The CSV header matching [`SpectrumRecord::to_csv_row`].
    pub const CSV_HEADER: &'static str = "k_over_pivot,k_internal_over_m_pl,N,k_over_aH,P_11,P_22,Re_P_12,Im_P_12,P_Qsigma,P_Qs,C_Qsigma_Qs,P_R,P_S,C_RS,r_C";

    /// Formats the row as comma-separated values.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e}",
            self.k_over_pivot,
            self.k_internal_over_m_pl,
            self.e_folds,
            self.k_over_ah,
            self.p_11,
            self.p_22,
            self.re_p_12,
            self.im_p_12,
            self.p_q_sigma,
            self.p_q_s,
            self.c_q_sigma_s,
            self.p_r,
            self.p_s,
            self.c_rs,
            self.r_c
        )
    }
}

/// One row of a single mode's evolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeEvolutionRecord {
    /// `k / k_pivot`.
    pub k_over_pivot: f64,
    /// `N = ln(a / a_initial)`.
    pub e_folds: f64,
    /// `k / (a H)`.
    pub k_over_ah: f64,
    /// `P_R`.
    pub p_r: f64,
    /// `P_S`.
    pub p_s: f64,
    /// Signed `C_RS`.
    pub c_rs: f64,
    /// Dimensionless signed turn rate.
    pub eta_perp: f64,
}

impl ModeEvolutionRecord {
    /// The CSV header matching [`ModeEvolutionRecord::to_csv_row`].
    pub const CSV_HEADER: &'static str = "k_over_pivot,N,k_over_aH,P_R,P_S,C_RS,eta_perp";

    /// Formats the row as comma-separated values.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            self.k_over_pivot,
            self.e_folds,
            self.k_over_ah,
            self.p_r,
            self.p_s,
            self.c_rs,
            self.eta_perp
        )
    }
}

/// The rows produced by one spectrum run.
#[derive(Clone, Debug)]
pub struct SpectrumRun {
    /// The realized pivot convention.
    pub pivot: Pivot,
    /// Background rows from the survey stage.
    pub background: Vec<BackgroundRecord>,
    /// One spectrum row per requested wavenumber.
    pub spectra: Vec<SpectrumRecord>,
    /// One spectrum row per wavenumber, evaluated at `k = a H`.
    pub horizon_exit_spectra: Vec<SpectrumRecord>,
    /// Sampled mode evolution for every wavenumber.
    pub mode_evolution: Vec<ModeEvolutionRecord>,
}

impl SpectrumRun {
    /// Writes the background rows as CSV.
    pub fn write_background_csv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_rows(
            writer,
            BackgroundRecord::CSV_HEADER,
            &self.background,
            |row| row.to_csv_row(),
        )
    }

    /// Writes the spectrum rows as CSV.
    pub fn write_spectra_csv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_rows(writer, SpectrumRecord::CSV_HEADER, &self.spectra, |row| {
            row.to_csv_row()
        })
    }

    /// Writes the horizon-exit spectrum rows as CSV.
    pub fn write_horizon_exit_spectra_csv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_rows(
            writer,
            SpectrumRecord::CSV_HEADER,
            &self.horizon_exit_spectra,
            |row| row.to_csv_row(),
        )
    }

    /// Writes the mode-evolution rows as CSV.
    pub fn write_mode_evolution_csv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_rows(
            writer,
            ModeEvolutionRecord::CSV_HEADER,
            &self.mode_evolution,
            |row| row.to_csv_row(),
        )
    }
}

fn write_rows<W: Write, R>(
    writer: &mut W,
    header: &str,
    rows: &[R],
    format: impl Fn(&R) -> String,
) -> io::Result<()> {
    writeln!(writer, "{header}")?;
    for row in rows {
        writeln!(writer, "{}", format(row))?;
    }
    Ok(())
}

/// Writes the metadata comment block that makes a run reproducible.
///
/// The revision is supplied by the caller, because the library does not read
/// the environment or shell out.
pub fn write_metadata<W: Write>(
    writer: &mut W,
    settings: &SpectrumRunSettings,
    pivot: &Pivot,
    revision: Option<&str>,
) -> io::Result<()> {
    writeln!(
        writer,
        "# generator: phys-type {}",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(
        writer,
        "# background_strategy: shared_cubic_hermite_in_e_folds"
    )?;
    if let Some(revision) = revision {
        writeln!(writer, "# revision: {revision}")?;
    }
    writeln!(
        writer,
        "# e_folds_before_exit: {:e}",
        settings.e_folds_before_exit
    )?;
    writeln!(writer, "# x_init: {:e}", start_depth(settings))?;
    match settings.end_of_inflation {
        EndOfInflation::SlowRollViolation => {
            writeln!(writer, "# end_of_inflation: epsilon_H = 1")?;
        }
        EndOfInflation::FieldValue { field, value } => writeln!(
            writer,
            "# end_of_inflation: field {field} crosses {:e} M_pl",
            value.into_value()
        )?,
    }
    // The absolute tolerances are per component group, so the metadata
    // records each of them rather than one number.
    let absolute = &settings.tolerances.absolute;
    writeln!(
        writer,
        "# absolute_tolerance_phi: {:e}",
        absolute.background.phi[0].into_value()
    )?;
    writeln!(
        writer,
        "# absolute_tolerance_pi: {:e}",
        absolute.background.pi[0].into_value()
    )?;
    writeln!(
        writer,
        "# absolute_tolerance_log_a: {:e}",
        absolute.background.log_scale_factor.into_value()
    )?;
    writeln!(
        writer,
        "# absolute_tolerance_mode_amplitude: {:e}",
        absolute.rescaled_amplitude.into_value()
    )?;
    writeln!(
        writer,
        "# absolute_tolerance_mode_velocity: {:e}",
        absolute.rescaled_velocity.into_value()
    )?;
    writeln!(
        writer,
        "# relative_tolerance: {:e}",
        settings.tolerances.relative
    )?;
    writeln!(
        writer,
        "# pivot: k_pivot = a H at N_end - {:e}",
        settings.pivot_e_folds_before_end
    )?;
    writeln!(
        writer,
        "# k_pivot_internal_over_m_pl: {:e}",
        pivot.k.into_value()
    )?;
    writeln!(
        writer,
        "# hubble_at_pivot_over_m_pl: {:e}",
        pivot.hubble.into_value()
    )?;
    writeln!(writer, "# epsilon_h_at_pivot: {:e}", pivot.epsilon_h)?;
    writeln!(writer, "# e_folds_at_end: {:e}", pivot.e_folds_at_end)?;
    writeln!(writer, "# e_folds_at_pivot: {:e}", pivot.e_folds_at_pivot)?;
    writeln!(
        writer,
        "# output_e_folds_before_end: {:e}",
        settings.output_e_folds_before_end
    )
}

/// Advances `system` until `event` stops being positive.
///
/// The crossing is refined inside the last accepted step, so the returned
/// state sits just on the positive side of the event. `on_accepted` observes
/// every accepted step, including the refined final one.
struct AdvanceControl<'a, A> {
    settings: &'a SpectrumRunSettings,
    tolerances: &'a ErrorTolerances<A>,
    stage: SpectrumRunStage,
}

fn advance_until<S, Event, Observe>(
    system: &S,
    initial_state: S::State,
    initial_time: Time,
    control: AdvanceControl<'_, S::AbsoluteTolerance>,
    mut event: Event,
    mut on_accepted: Observe,
) -> Result<(S::State, Time), SpectrumRunError>
where
    S: AdaptiveSystem<IndependentVariable = Time>,
    S::State: Clone,
    Event: FnMut(&S::State) -> f64,
    Observe: FnMut(&S::State, Time),
{
    let settings = control.settings;
    let tolerances = control.tolerances;
    let stage = control.stage;
    let limits = StepSizeLimits {
        min: settings.min_dt,
        max: settings.max_dt,
    };
    let mut state = initial_state;
    let mut time = initial_time;
    let mut dt = settings.initial_dt;

    if event(&state) <= 0.0 {
        return Ok((state, time));
    }

    for _ in 0..settings.max_steps {
        let attempt = adaptive_step(system, time, &state, dt, tolerances);

        if attempt.accepted {
            if event(&attempt.state) <= 0.0 {
                let (refined, refined_dt) = refine_crossing(system, time, &state, dt, &mut event);
                let refined_time = time + refined_dt;
                on_accepted(&refined, refined_time);

                return Ok((refined, refined_time));
            }

            state = attempt.state;
            time = time + dt;
            on_accepted(&state, time);
        } else if dt <= settings.min_dt {
            return Err(SpectrumRunError::MinimumStepReached { stage });
        }

        dt = next_step_size(dt, attempt.error_ratio, &limits);
    }

    Err(SpectrumRunError::StepLimitReached { stage })
}

/// Bisects inside one accepted step for the largest sub-step that keeps the
/// event positive.
///
/// Taking the positive side means a state handed to Bunch-Davies
/// initialization is never a hair shallower than the requested depth.
///
/// The sub-steps are taken with the same fifth-order method the step was
/// accepted on. A fourth-order sub-step would be cheaper, but the state it
/// returns is the one that starts a mode and the one the spectra are read
/// from, and its extra error is comparable to the systematic the
/// initialization depth is chosen to control.
fn refine_crossing<S, Event>(
    system: &S,
    time: Time,
    state: &S::State,
    dt: Time,
    event: &mut Event,
) -> (S::State, Time)
where
    S: AdaptiveSystem<IndependentVariable = Time>,
    S::State: Clone,
    Event: FnMut(&S::State) -> f64,
{
    let mut low = 0.0;
    let mut high = 1.0;
    let mut best = state.clone();
    let mut best_fraction = 0.0;

    for _ in 0..60 {
        let middle = 0.5 * (low + high);
        let trial = rk45_step(system, time, state, dt * middle).state;

        if event(&trial) > 0.0 {
            low = middle;
            best = trial;
            best_fraction = middle;
        } else {
            high = middle;
        }

        if high - low < 1.0e-14 {
            break;
        }
    }

    (best, dt * best_fraction)
}

/// Conformal-time counterpart of [`advance_until`].
///
/// Public settings remain expressed as local cosmic-time scales.  At every
/// attempt the proposed interval is converted with `d eta = dt / a`.  This
/// keeps the existing, physically readable controller limits while the ODE
/// itself sees a distinct conformal-time coordinate.
#[cfg(test)]
fn advance_conformal_until<S, Event, Observe, Scale>(
    system: &S,
    initial_state: S::State,
    control: AdvanceControl<'_, S::AbsoluteTolerance>,
    mut event: Event,
    mut on_accepted: Observe,
    scale_factor: Scale,
) -> Result<(S::State, ConformalTime), SpectrumRunError>
where
    S: AdaptiveSystem<IndependentVariable = ConformalTime>,
    S::State: Clone,
    Event: FnMut(&S::State) -> f64,
    Observe: FnMut(&S::State, ConformalTime),
    Scale: Fn(&S::State) -> ScaleFactor,
{
    let settings = control.settings;
    let tolerances = control.tolerances;
    let stage = control.stage;
    let limits = StepSizeLimits {
        min: settings.min_dt,
        max: settings.max_dt,
    };
    let mut state = initial_state;
    let mut conformal_time = ConformalTime::new(0.0);
    let mut dt = settings.initial_dt;

    if event(&state) <= 0.0 {
        return Ok((state, conformal_time));
    }

    for _ in 0..settings.max_steps {
        let a = scale_factor(&state).into_value();
        let deta = ConformalTime::new(dt.into_value() / a);
        let attempt = adaptive_step(system, conformal_time, &state, deta, tolerances);

        if attempt.accepted {
            if event(&attempt.state) <= 0.0 {
                let (refined, refined_deta) =
                    refine_conformal_crossing(system, conformal_time, &state, deta, &mut event);
                let refined_time = conformal_time + refined_deta;
                on_accepted(&refined, refined_time);
                return Ok((refined, refined_time));
            }

            state = attempt.state;
            conformal_time = conformal_time + deta;
            on_accepted(&state, conformal_time);
        } else if dt <= settings.min_dt {
            return Err(SpectrumRunError::MinimumStepReached { stage });
        }

        dt = next_step_size(dt, attempt.error_ratio, &limits);
    }

    Err(SpectrumRunError::StepLimitReached { stage })
}

#[cfg(test)]
fn refine_conformal_crossing<S, Event>(
    system: &S,
    conformal_time: ConformalTime,
    state: &S::State,
    deta: ConformalTime,
    event: &mut Event,
) -> (S::State, ConformalTime)
where
    S: AdaptiveSystem<IndependentVariable = ConformalTime>,
    S::State: Clone,
    Event: FnMut(&S::State) -> f64,
{
    let mut low = 0.0;
    let mut high = 1.0;
    let mut best = state.clone();
    let mut best_fraction = 0.0;

    for _ in 0..60 {
        let middle = 0.5 * (low + high);
        let trial = rk45_step(system, conformal_time, state, deta * middle).state;

        if event(&trial) > 0.0 {
            low = middle;
            best = trial;
            best_fraction = middle;
        } else {
            high = middle;
        }
        if high - low < 1.0e-14 {
            break;
        }
    }

    (best, deta * best_fraction)
}

#[derive(Clone, Copy, Debug)]
struct BackgroundCheckpoint<const N: usize> {
    e_folds: f64,
    state: MultifieldState<N>,
    dphi_dn: [crate::Field; N],
    dpi_dn: [crate::FieldVelocity; N],
}

/// A surveyed background, interpolated between accepted integration steps.
///
/// Elapsed e-folds are counted from the survey's initial state. The
/// interpolation is the cubic Hermite scheme of
/// `docs/background-checkpoint-interpolation.md`.
#[derive(Clone, Debug)]
pub struct BackgroundTrajectory<const N: usize = 2> {
    initial: MultifieldState<N>,
    checkpoints: Vec<BackgroundCheckpoint<N>>,
}

impl<const N: usize> BackgroundTrajectory<N> {
    /// The interpolated state `e_folds` after the initial state, clamped to
    /// the surveyed range.
    pub fn state_at(&self, e_folds: f64) -> MultifieldState<N> {
        let last = self.checkpoints.len() - 1;
        let upper = self
            .checkpoints
            .partition_point(|checkpoint| checkpoint.e_folds < e_folds)
            .clamp(1, last);
        let left = self.checkpoints[upper - 1];
        let right = self.checkpoints[upper];
        let width = right.e_folds - left.e_folds;
        let fraction = ((e_folds - left.e_folds) / width).clamp(0.0, 1.0);
        let squared = fraction * fraction;
        let cubed = squared * fraction;
        let h00 = 2.0 * cubed - 3.0 * squared + 1.0;
        let h10 = cubed - 2.0 * squared + fraction;
        let h01 = -2.0 * cubed + 3.0 * squared;
        let h11 = cubed - squared;

        MultifieldState::new(
            core::array::from_fn(|index| {
                left.state.phi[index] * h00
                    + left.dphi_dn[index] * (width * h10)
                    + right.state.phi[index] * h01
                    + right.dphi_dn[index] * (width * h11)
            }),
            core::array::from_fn(|index| {
                left.state.pi[index] * h00
                    + left.dpi_dn[index] * (width * h10)
                    + right.state.pi[index] * h01
                    + right.dpi_dn[index] * (width * h11)
            }),
            ScaleFactor::new((self.initial.a.into_value().ln() + e_folds).exp()),
        )
    }

    /// Elapsed e-folds of the checkpoints, between which the interpolated
    /// state is a cubic polynomial and only `C^1` across them.
    pub(crate) fn checkpoint_e_folds(&self) -> impl Iterator<Item = f64> + '_ {
        self.checkpoints.iter().map(|checkpoint| checkpoint.e_folds)
    }

    /// The first elapsed e-fold at which `event` falls from positive to
    /// zero or below, refined by bisection to `1e-13` e-folds and returned on
    /// the positive side; `None` if it never does.
    pub fn crossing(&self, mut event: impl FnMut(&MultifieldState<N>) -> f64) -> Option<f64> {
        let mut left = self.checkpoints[0].e_folds;
        let left_value = event(&self.checkpoints[0].state);
        if left_value <= 0.0 {
            return Some(left);
        }

        for checkpoint in &self.checkpoints[1..] {
            let right = checkpoint.e_folds;
            let right_value = event(&checkpoint.state);
            if right_value <= 0.0 {
                let mut low = left;
                let mut high = right;
                for _ in 0..60 {
                    let middle = 0.5 * (low + high);
                    if event(&self.state_at(middle)) > 0.0 {
                        low = middle;
                    } else {
                        high = middle;
                    }
                    if high - low <= 1.0e-13 {
                        break;
                    }
                }
                // Preserve the positive side, matching `refine_crossing` and
                // keeping Bunch-Davies initialization at least as deep as
                // requested.
                return Some(low);
            }
            left = right;
        }

        None
    }
}

impl<const N: usize> crate::BackgroundProvider<N> for BackgroundTrajectory<N> {
    fn state_at_e_folds(&self, e_folds: f64) -> MultifieldState<N> {
        self.state_at(e_folds)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ModesInEFolds<const N: usize>(Modes<N>);

#[derive(Clone, Copy, Debug, PartialEq)]
struct ModeDeltaInEFolds<const N: usize> {
    amplitude: [[ModeAmplitude; N]; N],
    velocity: [[ModeVelocity; N]; N],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ModeDerivativeInEFolds<const N: usize> {
    amplitude: [[ModeAmplitude; N]; N],
    velocity: [[ModeVelocity; N]; N],
}

impl<const N: usize> core::ops::Add<ModeDeltaInEFolds<N>> for ModesInEFolds<N> {
    type Output = Self;

    fn add(self, rhs: ModeDeltaInEFolds<N>) -> Self::Output {
        Self(Modes {
            rescaled_amplitude: core::array::from_fn(|field| {
                core::array::from_fn(|mode| {
                    self.0.rescaled_amplitude[field][mode] + rhs.amplitude[field][mode]
                })
            }),
            rescaled_velocity: core::array::from_fn(|field| {
                core::array::from_fn(|mode| {
                    self.0.rescaled_velocity[field][mode] + rhs.velocity[field][mode]
                })
            }),
        })
    }
}

impl<const N: usize> core::ops::Add for ModeDerivativeInEFolds<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            amplitude: core::array::from_fn(|field| {
                core::array::from_fn(|mode| {
                    self.amplitude[field][mode] + rhs.amplitude[field][mode]
                })
            }),
            velocity: core::array::from_fn(|field| {
                core::array::from_fn(|mode| self.velocity[field][mode] + rhs.velocity[field][mode])
            }),
        }
    }
}

impl<const N: usize> core::ops::Mul<f64> for ModeDerivativeInEFolds<N> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            amplitude: self.amplitude.map(|row| row.map(|entry| entry * rhs)),
            velocity: self.velocity.map(|row| row.map(|entry| entry * rhs)),
        }
    }
}

impl<const N: usize> core::ops::Mul<crate::Dimensionless> for ModeDerivativeInEFolds<N> {
    type Output = ModeDeltaInEFolds<N>;

    fn mul(self, rhs: crate::Dimensionless) -> Self::Output {
        let rhs = rhs.into_value();
        ModeDeltaInEFolds {
            amplitude: self.amplitude.map(|row| row.map(|entry| entry * rhs)),
            velocity: self.velocity.map(|row| row.map(|entry| entry * rhs)),
        }
    }
}

struct InterpolatedModeSystem<'a, P, G, const N: usize> {
    background: &'a CovariantMultifieldSystem<P, G, N>,
    trajectory: &'a BackgroundTrajectory<N>,
    k: Mass,
    /// Evolve a tensor polarization instead of the scalar modes: the
    /// equation of a massless field in flat field space, so the mass
    /// matrix and the connection terms drop out. Only entry `[0][0]` is used.
    tensor: bool,
}

impl<P, G, const N: usize> System for InterpolatedModeSystem<'_, P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    type IndependentVariable = crate::Dimensionless;
    type State = ModesInEFolds<N>;
    type Delta = ModeDeltaInEFolds<N>;
    type Derivative = ModeDerivativeInEFolds<N>;

    fn derivative(&self, e_folds: crate::Dimensionless, state: &Self::State) -> Self::Derivative {
        let background = self.trajectory.state_at(e_folds.into_value());
        let scale_factor = background.a;
        let hubble = self.background.hubble(&background);
        let connection = self.background.geometry.christoffel(&background.phi);
        let mass = self.background.perturbation_mass_matrix(&background);
        // Tensor modes: no mass matrix and no field-space connection.
        let coupling = if self.tensor { 0.0 } else { 1.0 };
        let conformal_field_velocity = background.pi.map(|entry| entry * scale_factor);
        let scale_factor_squared = scale_factor * scale_factor;
        let expansion =
            hubble * hubble * scale_factor_squared * (2.0 - self.background.epsilon_h(&background));
        let wave_number_squared = self.k * self.k;
        let a_h = scale_factor * hubble;

        let amplitude = core::array::from_fn(|upper| {
            core::array::from_fn(|mode| {
                let connection_term = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(
                        ModeVelocity::new(Complex64::new(0.0, 0.0)),
                        |sum, (first, second)| {
                            sum + connection.components()[upper][first][second]
                                * conformal_field_velocity[first]
                                * state.0.rescaled_amplitude[second][mode]
                        },
                    );
                (state.0.rescaled_velocity[upper][mode] - connection_term * coupling) / a_h
            })
        });
        let velocity = core::array::from_fn(|upper| {
            core::array::from_fn(|mode| {
                let restoring = (0..N).fold(
                    ModeAcceleration::new(Complex64::new(0.0, 0.0)),
                    |sum, lower| {
                        let entry = if upper == lower {
                            mass.components()[upper][lower] * scale_factor_squared * coupling
                                + wave_number_squared
                                - expansion
                        } else {
                            mass.components()[upper][lower] * scale_factor_squared * coupling
                        };
                        sum + entry * state.0.rescaled_amplitude[lower][mode]
                    },
                );
                let connection_term = (0..N)
                    .flat_map(|first| (0..N).map(move |second| (first, second)))
                    .fold(
                        ModeAcceleration::new(Complex64::new(0.0, 0.0)),
                        |sum, (first, second)| {
                            sum + connection.components()[upper][first][second]
                                * conformal_field_velocity[first]
                                * state.0.rescaled_velocity[second][mode]
                        },
                    );
                (-restoring - connection_term * coupling) / a_h
            })
        });

        ModeDerivativeInEFolds {
            amplitude,
            velocity,
        }
    }
}

#[derive(Clone, Copy)]
struct ModeTolerancesInEFolds {
    amplitude: ModeAmplitude<f64>,
    velocity: ModeVelocity<f64>,
}

impl<P, G, const N: usize> AdaptiveSystem for InterpolatedModeSystem<'_, P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
{
    type AbsoluteTolerance = ModeTolerancesInEFolds;

    fn error_ratio(
        &self,
        current: &Self::State,
        next: &Self::State,
        error: &Self::Delta,
        tolerances: &ErrorTolerances<Self::AbsoluteTolerance>,
    ) -> f64 {
        let amplitude_scale = mode_scale(&current.0.rescaled_amplitude, &next.0.rescaled_amplitude);
        let velocity_scale = mode_scale(&current.0.rescaled_velocity, &next.0.rescaled_velocity);

        (0..N)
            .flat_map(|field| (0..N).map(move |mode| (field, mode)))
            .map(|(field, mode)| {
                crate::perturbation::normalized_mode_error(
                    error.amplitude[field][mode],
                    amplitude_scale,
                    tolerances.absolute.amplitude,
                    tolerances.relative,
                )
                .max(crate::perturbation::normalized_mode_error(
                    error.velocity[field][mode],
                    velocity_scale,
                    tolerances.absolute.velocity,
                    tolerances.relative,
                ))
            })
            .fold(0.0, f64::max)
    }
}

fn mode_scale<D: Copy, const N: usize>(
    current: &[[crate::Quantity<D, Complex64>; N]; N],
    next: &[[crate::Quantity<D, Complex64>; N]; N],
) -> crate::Quantity<D, f64> {
    crate::Quantity::new(
        current
            .iter()
            .chain(next.iter())
            .flatten()
            .map(ComplexMagnitude::complex_magnitude)
            .fold(0.0, f64::max),
    )
}

fn advance_interpolated_modes<P, G, Observe, const N: usize>(
    system: &InterpolatedModeSystem<'_, P, G, N>,
    initial_state: ModesInEFolds<N>,
    initial_e_folds: f64,
    target_e_folds: f64,
    settings: &SpectrumRunSettings,
    stage: SpectrumRunStage,
    mut on_accepted: Observe,
) -> Result<ModesInEFolds<N>, SpectrumRunError>
where
    P: TwiceDifferentiableMultifieldPotential<N>,
    G: CurvedFieldSpaceGeometry<N>,
    Observe: FnMut(&ModesInEFolds<N>, f64),
{
    let limits = StepSizeLimits {
        min: settings.min_dt,
        max: settings.max_dt,
    };
    let tolerances = ErrorTolerances::new(
        ModeTolerancesInEFolds {
            amplitude: settings.tolerances.absolute.rescaled_amplitude,
            velocity: settings.tolerances.absolute.rescaled_velocity,
        },
        settings.tolerances.relative,
    );
    let mut state = initial_state;
    let mut e_folds = initial_e_folds;
    let mut dt = settings.initial_dt;

    for _ in 0..settings.max_steps {
        let remaining = target_e_folds - e_folds;
        if remaining <= 1.0e-13 {
            return Ok(state);
        }
        let background = system.trajectory.state_at(e_folds);
        let hubble = system.background.hubble(&background).into_value();
        let proposed_dn = hubble * dt.into_value();
        let dn = proposed_dn.min(remaining);
        let effective_dt = Time::new(dn / hubble);
        let attempt = adaptive_step(
            system,
            crate::Dimensionless::new(e_folds),
            &state,
            crate::Dimensionless::new(dn),
            &tolerances,
        );

        if attempt.accepted {
            state = attempt.state;
            e_folds += dn;
            on_accepted(&state, e_folds);
        } else if effective_dt <= settings.min_dt {
            return Err(SpectrumRunError::MinimumStepReached { stage });
        }

        dt = next_step_size(effective_dt, attempt.error_ratio, &limits);
    }

    Err(SpectrumRunError::StepLimitReached { stage })
}

impl<P, G, const N: usize> CovariantMultifieldSystem<P, G, N>
where
    P: MultifieldPotential<N>,
    G: FieldSpaceGeometry<N>,
{
    fn background_checkpoint(
        &self,
        initial: &MultifieldState<N>,
        state: &MultifieldState<N>,
    ) -> BackgroundCheckpoint<N> {
        let hubble = self.hubble(state);
        let derivative = self.derivative(Time::new(0.0), state);

        BackgroundCheckpoint {
            e_folds: state.e_folds(initial),
            state: *state,
            dphi_dn: derivative.dphi_dt.map(|entry| entry / hubble),
            dpi_dn: derivative.dpi_dt.map(|entry| entry / hubble),
        }
    }

    /// Integrates the background to the end-of-inflation event, checking the
    /// geometry's domain on every accepted step. `on_accepted` sees every
    /// accepted state and the number of accepted steps so far.
    pub(crate) fn survey_trajectory(
        &self,
        initial: MultifieldState<N>,
        settings: &SpectrumRunSettings,
        mut on_accepted: impl FnMut(&MultifieldState<N>, usize),
    ) -> Result<(BackgroundTrajectory<N>, f64, MultifieldState<N>), SpectrumRunError> {
        self.geometry.check_domain(&initial.phi).map_err(|error| {
            SpectrumRunError::OutsideGeometryDomain {
                e_folds: 0.0,
                error,
            }
        })?;
        let mut domain_error = None;
        let mut accepted = 0_usize;
        let mut checkpoints = vec![self.background_checkpoint(&initial, &initial)];

        // Inflation ends at the requested event. The survey also stops if the
        // e-folding budget runs out, or if slow roll is violated before a
        // field-value event; both are reported separately.
        let field_event = match settings.end_of_inflation {
            EndOfInflation::SlowRollViolation => None,
            EndOfInflation::FieldValue { field, value } => {
                let side = (initial.phi[field] - value).into_value().signum();
                Some((field, value, side))
            }
        };
        let tolerances = background_tolerances::<N>(settings);
        let control = || AdvanceControl {
            settings,
            tolerances: &tolerances,
            stage: SpectrumRunStage::BackgroundSurvey,
        };
        let mut record = |state: &MultifieldState<N>| {
            accepted += 1;
            checkpoints.push(self.background_checkpoint(&initial, state));
            if domain_error.is_none()
                && let Err(error) = self.geometry.check_domain(&state.phi)
            {
                domain_error = Some(SpectrumRunError::OutsideGeometryDomain {
                    e_folds: state.e_folds(&initial),
                    error,
                });
            }
            on_accepted(state, accepted);
        };
        let (final_state, time) = advance_until(
            self,
            initial,
            Time::new(0.0),
            control(),
            |state| {
                let remaining = settings.max_e_folds - state.e_folds(&initial);
                let inflating = 1.0 - self.epsilon_h(state);
                let before_field_event = field_event
                    .map_or(f64::INFINITY, |(field, value, side)| {
                        side * (state.phi[field] - value).into_value()
                    });

                remaining.min(inflating).min(before_field_event)
            },
            |state, _| record(state),
        )?;

        let e_folds = final_state.e_folds(&initial);
        let ended = match field_event {
            None => self.epsilon_h(&final_state) >= 1.0 - 1.0e-9,
            Some((field, value, side)) => {
                let distance = side * (final_state.phi[field] - value).into_value();
                if distance > 1.0e-9 * value.into_value().abs().max(1.0) {
                    if self.epsilon_h(&final_state) >= 1.0 - 1.0e-9 {
                        return Err(SpectrumRunError::SlowRollEndedFirst { e_folds });
                    }
                    false
                } else {
                    true
                }
            }
        };
        if !ended {
            return Err(SpectrumRunError::InflationDidNotEnd { e_folds });
        }
        let final_state = if settings.e_folds_after_end > 0.0 {
            let stop = e_folds + settings.e_folds_after_end;
            advance_until(
                self,
                final_state,
                time,
                control(),
                |state| stop - state.e_folds(&initial),
                |state, _| record(state),
            )?
            .0
        } else {
            final_state
        };

        if let Some(error) = domain_error {
            return Err(error);
        }
        Ok((
            BackgroundTrajectory {
                initial,
                checkpoints,
            },
            e_folds,
            final_state,
        ))
    }

    fn background_sample(
        &self,
        initial: &MultifieldState<N>,
        state: &MultifieldState<N>,
    ) -> BackgroundSample<N> {
        BackgroundSample {
            e_folds: state.e_folds(initial),
            phi: state.phi.map(Quantity::into_value),
            pi: state.pi.map(Quantity::into_value),
            hubble: self.hubble(state).into_value(),
            epsilon_h: self.epsilon_h(state),
            eta_perp: if N == 1 {
                0.0
            } else {
                self.kinematics(state)
                    .map_or(f64::NAN, |kinematics| kinematics.eta_perp.into_value())
            },
        }
    }

    /// Integrates the background from `initial` to the end-of-inflation event
    /// of `settings`, for any number of fields, checking the geometry's
    /// domain on every accepted step.
    pub fn survey(
        &self,
        initial: MultifieldState<N>,
        settings: &SpectrumRunSettings,
    ) -> Result<BackgroundSurvey<N>, SpectrumRunError> {
        validate(settings)?;
        let mut samples = vec![self.background_sample(&initial, &initial)];
        let (trajectory, e_folds_at_end, final_state) =
            self.survey_trajectory(initial, settings, |state, accepted| {
                if accepted.is_multiple_of(settings.background_sample_every) {
                    samples.push(self.background_sample(&initial, state));
                }
            })?;
        samples.push(self.background_sample(&initial, &final_state));
        Ok(BackgroundSurvey {
            samples,
            trajectory,
            e_folds_at_end,
        })
    }
}

/// The spectra of one mode on a surveyed trajectory, for any number of
/// fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeSpectrum<const N: usize> {
    /// Comoving wavenumber in Planck units.
    pub k: f64,
    /// Elapsed e-folds of the output slice.
    pub e_folds: f64,
    /// `k/(aH)` on the output slice.
    pub k_over_ah: f64,
    /// Curvature and basis-free entropy spectra.
    pub spectra: crate::InvariantSpectra<N>,
}

impl<P, G, const N: usize> CovariantMultifieldSystem<P, G, N>
where
    P: TwiceDifferentiableMultifieldPotential<N> + Clone,
    G: CurvedFieldSpaceGeometry<N> + Clone,
{
    /// Evolves the mode of wavenumber `k` on a surveyed `trajectory` from its
    /// Bunch-Davies start, `settings.e_folds_before_exit` e-folds before
    /// horizon exit, to `e_folds_at_output`, and projects it onto the
    /// curvature and basis-free entropy spectra.
    pub fn mode_spectrum(
        &self,
        settings: &SpectrumRunSettings,
        trajectory: &BackgroundTrajectory<N>,
        k: Mass,
        e_folds_at_output: f64,
    ) -> Result<ModeSpectrum<N>, SpectrumRunError> {
        validate(settings)?;
        let start_e_folds = trajectory
            .crossing(|state| {
                (k / (state.a * self.hubble(state))).into_value() - start_depth(settings)
            })
            .ok_or(SpectrumRunError::StepLimitReached {
                stage: SpectrumRunStage::ModeStart { k: k.into_value() },
            })?;
        let conformal_system = ConformalPerturbationSystem::new(self.clone(), k);
        let initialized = conformal_system
            .bunch_davies_state(trajectory.state_at(start_e_folds), start_depth(settings))
            .map_err(|source| SpectrumRunError::ModeInitialization {
                k: k.into_value(),
                source,
            })?;
        let mode_system = InterpolatedModeSystem {
            background: self,
            trajectory,
            k,
            tensor: false,
        };
        let state = advance_interpolated_modes(
            &mode_system,
            ModesInEFolds(initialized.modes),
            start_e_folds,
            e_folds_at_output,
            settings,
            SpectrumRunStage::ModeEvolution { k: k.into_value() },
            |_, _| {},
        )?;
        let background = trajectory.state_at(e_folds_at_output);
        let k_over_ah = (k / (background.a * self.hubble(&background))).into_value();
        if k_over_ah > 1.0 {
            return Err(SpectrumRunError::OutputBeforeHorizonExit {
                k: k.into_value(),
                k_over_ah,
            });
        }
        let cosmic = ConformalPerturbationState {
            background: ConformalBackgroundState::from_cosmic(background),
            modes: state.0,
        }
        .cosmic();
        let spectra = crate::PerturbationSystem::new(self.clone(), k)
            .invariant_spectra(&cosmic, settings.spectrum_tolerances)
            .map_err(|source| SpectrumRunError::Spectrum {
                k: k.into_value(),
                source,
            })?;
        Ok(ModeSpectrum {
            k: k.into_value(),
            e_folds: e_folds_at_output,
            k_over_ah,
            spectra,
        })
    }

    /// The tensor power spectrum `P_t` of wavenumber `k` at
    /// `e_folds_at_output`, summed over both polarizations.
    ///
    /// Each polarization obeys `v'' + (k^2 - a''/a) v = 0` for
    /// `v = a M_pl h / 2`, independent of the fields, and starts in the
    /// Bunch-Davies vacuum `settings.e_folds_before_exit` e-folds before
    /// horizon exit, like the scalar modes. Then
    /// `P_t = 8 (k^3 / 2 pi^2) |v / a|^2 / M_pl^2`, which is `2 H^2 / (pi^2 M_pl^2)`
    /// in de Sitter. `h_ij` is unchanged by a conformal transformation, so this
    /// is also the Jordan-frame tensor spectrum.
    pub fn tensor_spectrum(
        &self,
        settings: &SpectrumRunSettings,
        trajectory: &BackgroundTrajectory<N>,
        k: Mass,
        e_folds_at_output: f64,
    ) -> Result<f64, SpectrumRunError> {
        validate(settings)?;
        let start_e_folds = trajectory
            .crossing(|state| {
                (k / (state.a * self.hubble(state))).into_value() - start_depth(settings)
            })
            .ok_or(SpectrumRunError::StepLimitReached {
                stage: SpectrumRunStage::ModeStart { k: k.into_value() },
            })?;
        let amplitude: ModeAmplitude<f64> = crate::Dimensionless::new(1.0) / (k * 2.0).sqrt();
        let amplitude = ModeAmplitude::new(Complex64::new(amplitude.into_value(), 0.0));
        let frequency =
            crate::Quantity::<crate::M1, Complex64>::new(Complex64::new(0.0, -k.into_value()));
        let zero_amplitude = ModeAmplitude::new(Complex64::new(0.0, 0.0));
        let zero_velocity = ModeVelocity::new(Complex64::new(0.0, 0.0));
        let mut rescaled_amplitude = [[zero_amplitude; N]; N];
        let mut rescaled_velocity = [[zero_velocity; N]; N];
        rescaled_amplitude[0][0] = amplitude;
        rescaled_velocity[0][0] = amplitude * frequency;
        let mode_system = InterpolatedModeSystem {
            background: self,
            trajectory,
            k,
            tensor: true,
        };
        let state = advance_interpolated_modes(
            &mode_system,
            ModesInEFolds(Modes {
                rescaled_amplitude,
                rescaled_velocity,
            }),
            start_e_folds,
            e_folds_at_output,
            settings,
            SpectrumRunStage::ModeEvolution { k: k.into_value() },
            |_, _| {},
        )?;
        let scale_factor = trajectory.state_at(e_folds_at_output).a.into_value();
        let v = state.0.rescaled_amplitude[0][0].into_value() / scale_factor;
        let planck = self.m_pl.into_value();
        let k = k.into_value();
        Ok(
            8.0 * k * k * k / (2.0 * core::f64::consts::PI * core::f64::consts::PI) * v.norm_sqr()
                / (planck * planck),
        )
    }
}

impl<P, G> CovariantMultifieldSystem<P, G, 2>
where
    P: TwiceDifferentiableMultifieldPotential<2> + Clone,
    G: CurvedFieldSpaceGeometry<2> + Clone,
{
    /// Runs the background survey, fixes the pivot, and evolves one mode
    /// matrix per requested wavenumber.
    pub fn run_two_field_spectra(
        &self,
        initial: MultifieldState<2>,
        settings: &SpectrumRunSettings,
    ) -> Result<SpectrumRun, SpectrumRunError> {
        validate(settings)?;

        let (background, trajectory, e_folds_at_end) = self.survey_background(initial, settings)?;
        let e_folds_at_pivot = e_folds_at_end - settings.pivot_e_folds_before_end;
        let e_folds_at_output = e_folds_at_end - settings.output_e_folds_before_end;
        if e_folds_at_pivot < 0.0 || e_folds_at_output < 0.0 {
            return Err(SpectrumRunError::NotEnoughEFolds {
                available: e_folds_at_end,
                required: settings
                    .pivot_e_folds_before_end
                    .max(settings.output_e_folds_before_end),
            });
        }

        let pivot_state = trajectory.state_at(e_folds_at_pivot);
        let pivot_hubble = self.hubble(&pivot_state);
        let pivot = Pivot {
            k: pivot_state.a * pivot_hubble,
            hubble: pivot_hubble,
            epsilon_h: self.epsilon_h(&pivot_state),
            e_folds_at_end,
            e_folds_at_pivot,
        };

        let mut spectra = Vec::with_capacity(settings.k_count);
        let mut horizon_exit_spectra = Vec::with_capacity(settings.k_count);
        let mut mode_evolution = Vec::new();
        for index in 0..settings.k_count {
            let k_over_pivot = grid_point(settings, index);
            let (record, horizon_exit_record, rows) = self.evolve_one_mode(
                initial,
                settings,
                &trajectory,
                pivot.k * k_over_pivot,
                k_over_pivot,
                e_folds_at_output,
            )?;

            spectra.push(record);
            horizon_exit_spectra.push(horizon_exit_record);
            mode_evolution.extend(rows);
        }

        Ok(SpectrumRun {
            pivot,
            background,
            spectra,
            horizon_exit_spectra,
            mode_evolution,
        })
    }

    /// Integrates the background to the end of inflation, recording rows.
    pub(crate) fn survey_background(
        &self,
        initial: MultifieldState<2>,
        settings: &SpectrumRunSettings,
    ) -> Result<(Vec<BackgroundRecord>, BackgroundTrajectory, f64), SpectrumRunError> {
        self.geometry.check_domain(&initial.phi).map_err(|error| {
            SpectrumRunError::OutsideGeometryDomain {
                e_folds: 0.0,
                error,
            }
        })?;
        let initial_kinematics = self
            .oriented_kinematics(&initial, None)
            .map_err(SpectrumRunError::KinematicBasis)?;
        let mut rows = vec![self.background_record(&initial, &initial, &initial_kinematics)];
        let mut previous_normal = Some(initial_kinematics.basis.normal);
        let mut basis_error = None;
        let (trajectory, e_folds, final_state) =
            self.survey_trajectory(initial, settings, |state, accepted| {
                if basis_error.is_some() {
                    return;
                }
                match self.oriented_kinematics(state, previous_normal.as_ref()) {
                    Ok(kinematics) => {
                        previous_normal = Some(kinematics.basis.normal);
                        if accepted.is_multiple_of(settings.background_sample_every) {
                            rows.push(self.background_record(&initial, state, &kinematics));
                        }
                    }
                    Err(error) => basis_error = Some(error),
                }
            })?;
        if let Some(error) = basis_error {
            return Err(SpectrumRunError::KinematicBasis(error));
        }
        let final_kinematics = self
            .oriented_kinematics(&final_state, previous_normal.as_ref())
            .map_err(SpectrumRunError::KinematicBasis)?;
        rows.push(self.background_record(&initial, &final_state, &final_kinematics));

        Ok((rows, trajectory, e_folds))
    }

    fn background_record(
        &self,
        initial: &MultifieldState<2>,
        state: &MultifieldState<2>,
        kinematics: &TwoFieldKinematics,
    ) -> BackgroundRecord {
        let squared = self.m_pl * self.m_pl;

        BackgroundRecord {
            e_folds: state.e_folds(initial),
            phi: core::array::from_fn(|index| (state.phi[index] / self.m_pl).into_value()),
            pi: core::array::from_fn(|index| (state.pi[index] / squared).into_value()),
            hubble: (self.hubble(state) / self.m_pl).into_value(),
            epsilon_h: self.epsilon_h(state),
            e_sigma: component_pair(&kinematics.basis.tangent),
            e_s: component_pair(&kinematics.basis.normal),
            eta_perp: kinematics.eta_perp.into_value(),
        }
    }

    /// Carries one wavenumber against the stored background trajectory.
    fn evolve_one_mode(
        &self,
        initial: MultifieldState<2>,
        settings: &SpectrumRunSettings,
        trajectory: &BackgroundTrajectory,
        k: Mass,
        k_over_pivot: f64,
        e_folds_at_output: f64,
    ) -> Result<(SpectrumRecord, SpectrumRecord, Vec<ModeEvolutionRecord>), SpectrumRunError> {
        let conformal_system = ConformalPerturbationSystem::new(self.clone(), k);
        let start_e_folds = trajectory
            .crossing(|state| {
                (k / (state.a * self.hubble(state))).into_value() - start_depth(settings)
            })
            .ok_or(SpectrumRunError::StepLimitReached {
                stage: SpectrumRunStage::ModeStart { k: k.into_value() },
            })?;
        let horizon_e_folds = trajectory
            .crossing(|state| (k / (state.a * self.hubble(state))).into_value() - 1.0)
            .ok_or(SpectrumRunError::StepLimitReached {
                stage: SpectrumRunStage::HorizonExit { k: k.into_value() },
            })?;
        let start_background = trajectory.state_at(start_e_folds);
        let initialized = conformal_system
            .bunch_davies_state(start_background, start_depth(settings))
            .map_err(|source| SpectrumRunError::ModeInitialization {
                k: k.into_value(),
                source,
            })?;
        let mode_system = InterpolatedModeSystem {
            background: self,
            trajectory,
            k,
            tensor: false,
        };
        let mut state = ModesInEFolds(initialized.modes);
        let initial_combined = ConformalPerturbationState {
            background: ConformalBackgroundState::from_cosmic(start_background),
            modes: state.0,
        };
        let (projected, kinematics) =
            self.project_state(&conformal_system, &initial_combined, settings, None)?;
        let mut rows = vec![self.mode_record(
            &initial,
            &initial_combined,
            k,
            k_over_pivot,
            &projected,
            &kinematics,
        )];
        let mut previous_normal = Some(kinematics.basis.normal);
        let mut sample_error = None;
        let mut accepted = 0_usize;

        state = advance_interpolated_modes(
            &mode_system,
            state,
            start_e_folds,
            horizon_e_folds,
            settings,
            SpectrumRunStage::HorizonExit { k: k.into_value() },
            |modes, e_folds| {
                accepted += 1;
                if sample_error.is_some() || !accepted.is_multiple_of(settings.mode_sample_every) {
                    return;
                }
                let combined = ConformalPerturbationState {
                    background: ConformalBackgroundState::from_cosmic(trajectory.state_at(e_folds)),
                    modes: modes.0,
                };
                match self.project_state(
                    &conformal_system,
                    &combined,
                    settings,
                    previous_normal.as_ref(),
                ) {
                    Ok((projected, kinematics)) => {
                        previous_normal = Some(kinematics.basis.normal);
                        rows.push(self.mode_record(
                            &initial,
                            &combined,
                            k,
                            k_over_pivot,
                            &projected,
                            &kinematics,
                        ));
                    }
                    Err(error) => sample_error = Some(error),
                }
            },
        )?;
        if let Some(error) = sample_error {
            return Err(error);
        }

        let horizon_state = ConformalPerturbationState {
            background: ConformalBackgroundState::from_cosmic(trajectory.state_at(horizon_e_folds)),
            modes: state.0,
        };
        let (horizon_projected, horizon_kinematics) = self.project_state(
            &conformal_system,
            &horizon_state,
            settings,
            previous_normal.as_ref(),
        )?;
        previous_normal = Some(horizon_kinematics.basis.normal);
        rows.push(self.mode_record(
            &initial,
            &horizon_state,
            k,
            k_over_pivot,
            &horizon_projected,
            &horizon_kinematics,
        ));
        let horizon_exit_record = self.spectrum_record(
            &initial,
            &horizon_state,
            k,
            k_over_pivot,
            &horizon_projected,
            settings,
        )?;

        state = advance_interpolated_modes(
            &mode_system,
            state,
            horizon_e_folds,
            e_folds_at_output,
            settings,
            SpectrumRunStage::ModeEvolution { k: k.into_value() },
            |modes, e_folds| {
                accepted += 1;
                if sample_error.is_some() || !accepted.is_multiple_of(settings.mode_sample_every) {
                    return;
                }
                let combined = ConformalPerturbationState {
                    background: ConformalBackgroundState::from_cosmic(trajectory.state_at(e_folds)),
                    modes: modes.0,
                };
                match self.project_state(
                    &conformal_system,
                    &combined,
                    settings,
                    previous_normal.as_ref(),
                ) {
                    Ok((projected, kinematics)) => {
                        previous_normal = Some(kinematics.basis.normal);
                        rows.push(self.mode_record(
                            &initial,
                            &combined,
                            k,
                            k_over_pivot,
                            &projected,
                            &kinematics,
                        ));
                    }
                    Err(error) => sample_error = Some(error),
                }
            },
        )?;
        if let Some(error) = sample_error {
            return Err(error);
        }

        let final_background = trajectory.state_at(e_folds_at_output);
        let depth = (k / (final_background.a * self.hubble(&final_background))).into_value();
        if depth.is_nan() || depth >= 1.0 {
            return Err(SpectrumRunError::OutputBeforeHorizonExit {
                k: k.into_value(),
                k_over_ah: depth,
            });
        }
        let final_state = ConformalPerturbationState {
            background: ConformalBackgroundState::from_cosmic(final_background),
            modes: state.0,
        };
        let (projected, kinematics) = self.project_state(
            &conformal_system,
            &final_state,
            settings,
            previous_normal.as_ref(),
        )?;
        rows.push(self.mode_record(
            &initial,
            &final_state,
            k,
            k_over_pivot,
            &projected,
            &kinematics,
        ));
        let record = self.spectrum_record(
            &initial,
            &final_state,
            k,
            k_over_pivot,
            &projected,
            settings,
        )?;

        Ok((record, horizon_exit_record, rows))
    }

    /// Coupled background-plus-mode reference retained for regression tests.
    #[cfg(test)]
    fn evolve_one_mode_reference(
        &self,
        initial: MultifieldState<2>,
        settings: &SpectrumRunSettings,
        k: Mass,
        k_over_pivot: f64,
        e_folds_at_output: f64,
    ) -> Result<(SpectrumRecord, SpectrumRecord, Vec<ModeEvolutionRecord>), SpectrumRunError> {
        let system = ConformalPerturbationSystem::new(self.clone(), k);

        // Carry the background forward until this mode sits at x_init.
        let (start, _start_time) = advance_until(
            self,
            initial,
            Time::new(0.0),
            AdvanceControl {
                settings,
                tolerances: &background_tolerances(settings),
                stage: SpectrumRunStage::ModeStart { k: k.into_value() },
            },
            |state| (k / (state.a * self.hubble(state))).into_value() - start_depth(settings),
            |_, _| {},
        )?;

        let mode_state = system
            .bunch_davies_state(start, start_depth(settings))
            .map_err(|source| SpectrumRunError::ModeInitialization {
                k: k.into_value(),
                source,
            })?;

        let (projected, kinematics) = self.project_state(&system, &mode_state, settings, None)?;
        let mut rows = vec![self.mode_record(
            &initial,
            &mode_state,
            k,
            k_over_pivot,
            &projected,
            &kinematics,
        )];
        let mut previous_normal = Some(kinematics.basis.normal);
        let mut sample_error = None;
        let mut accepted = 0_usize;

        // Stop at this mode's own horizon exit first.  Keeping this as an
        // explicit refined event makes the exit and final spectra independent
        // products rather than asking callers to reconstruct one from sampled
        // evolution rows.
        let (horizon_exit_state, _) = advance_conformal_until(
            &system,
            mode_state,
            AdvanceControl {
                settings,
                tolerances: &settings.tolerances,
                stage: SpectrumRunStage::HorizonExit { k: k.into_value() },
            },
            |state| {
                let background = state.background.cosmic();
                (k / (background.a * self.hubble(&background))).into_value() - 1.0
            },
            |state, _| {
                accepted += 1;
                if sample_error.is_some() || !accepted.is_multiple_of(settings.mode_sample_every) {
                    return;
                }
                match self.project_state(&system, state, settings, previous_normal.as_ref()) {
                    Ok((projected, kinematics)) => {
                        previous_normal = Some(kinematics.basis.normal);
                        rows.push(self.mode_record(
                            &initial,
                            state,
                            k,
                            k_over_pivot,
                            &projected,
                            &kinematics,
                        ));
                    }
                    Err(error) => sample_error = Some(error),
                }
            },
            |state| state.background.cosmic().a,
        )?;

        if let Some(error) = sample_error {
            return Err(error);
        }

        let (horizon_projected, horizon_kinematics) = self.project_state(
            &system,
            &horizon_exit_state,
            settings,
            previous_normal.as_ref(),
        )?;
        previous_normal = Some(horizon_kinematics.basis.normal);
        rows.push(self.mode_record(
            &initial,
            &horizon_exit_state,
            k,
            k_over_pivot,
            &horizon_projected,
            &horizon_kinematics,
        ));
        let horizon_exit_record = self.spectrum_record(
            &initial,
            &horizon_exit_state,
            k,
            k_over_pivot,
            &horizon_projected,
            settings,
        )?;

        let (final_state, _) = advance_conformal_until(
            &system,
            horizon_exit_state,
            AdvanceControl {
                settings,
                tolerances: &settings.tolerances,
                stage: SpectrumRunStage::ModeEvolution { k: k.into_value() },
            },
            |state| e_folds_at_output - state.background.e_folds(&initial),
            |state, _| {
                accepted += 1;
                if sample_error.is_some() || !accepted.is_multiple_of(settings.mode_sample_every) {
                    return;
                }
                match self.project_state(&system, state, settings, previous_normal.as_ref()) {
                    Ok((projected, kinematics)) => {
                        previous_normal = Some(kinematics.basis.normal);
                        rows.push(self.mode_record(
                            &initial,
                            state,
                            k,
                            k_over_pivot,
                            &projected,
                            &kinematics,
                        ));
                    }
                    Err(error) => sample_error = Some(error),
                }
            },
            |state| state.background.cosmic().a,
        )?;

        if let Some(error) = sample_error {
            return Err(error);
        }

        // A mode that has not left the horizon by the output time still holds
        // its vacuum, which would be reported as a spectrum. The driver
        // refuses rather than returning a number that looks physical.
        let final_background = final_state.background.cosmic();
        let depth = (k / (final_background.a * self.hubble(&final_background))).into_value();
        // Written without a negated comparison so a NaN depth is refused
        // explicitly rather than by falling through a flipped inequality.
        if depth.is_nan() || depth >= 1.0 {
            return Err(SpectrumRunError::OutputBeforeHorizonExit {
                k: k.into_value(),
                k_over_ah: depth,
            });
        }

        let (projected, kinematics) =
            self.project_state(&system, &final_state, settings, previous_normal.as_ref())?;
        rows.push(self.mode_record(
            &initial,
            &final_state,
            k,
            k_over_pivot,
            &projected,
            &kinematics,
        ));

        let record = self.spectrum_record(
            &initial,
            &final_state,
            k,
            k_over_pivot,
            &projected,
            settings,
        )?;

        Ok((record, horizon_exit_record, rows))
    }

    fn mode_record(
        &self,
        initial: &MultifieldState<2>,
        state: &ConformalPerturbationState<2>,
        k: Mass,
        k_over_pivot: f64,
        projected: &ProjectedSpectra,
        kinematics: &TwoFieldKinematics,
    ) -> ModeEvolutionRecord {
        let background = state.background.cosmic();
        let hubble = self.hubble(&background);

        ModeEvolutionRecord {
            k_over_pivot,
            e_folds: state.background.e_folds(initial),
            k_over_ah: (k / (background.a * hubble)).into_value(),
            p_r: projected.curvature.into_value(),
            p_s: projected.entropy.into_value(),
            c_rs: projected.cross.into_value(),
            eta_perp: kinematics.eta_perp.into_value(),
        }
    }

    fn spectrum_record(
        &self,
        initial: &MultifieldState<2>,
        state: &ConformalPerturbationState<2>,
        k: Mass,
        k_over_pivot: f64,
        projected: &ProjectedSpectra,
        settings: &SpectrumRunSettings,
    ) -> Result<SpectrumRecord, SpectrumRunError> {
        let matrix = field_power_spectrum(k, &mode_covariance(&state.cosmic()));
        let components = matrix.components();
        let squared = (self.m_pl * self.m_pl).into_value();
        let cross = components[0][1];
        let background = state.background.cosmic();

        // The diagonal entries are real by construction.  Validate the
        // off-diagonal imaginary residual at every exported epoch before
        // splitting it into CSV columns.
        crate::spectra::discard_imaginary(
            cross,
            crate::spectra::spectrum_scale(&matrix),
            settings.spectrum_tolerances.imaginary_residual,
        )
        .map_err(|source| SpectrumRunError::Spectrum {
            k: k.into_value(),
            source,
        })?;

        Ok(SpectrumRecord {
            k_over_pivot,
            k_internal_over_m_pl: (k / self.m_pl).into_value(),
            e_folds: state.background.e_folds(initial),
            k_over_ah: (k / (background.a * self.hubble(&background))).into_value(),
            p_11: components[0][0].value().re / squared,
            p_22: components[1][1].value().re / squared,
            re_p_12: cross.value().re / squared,
            im_p_12: cross.value().im / squared,
            p_q_sigma: projected.field_adiabatic.into_value() / squared,
            p_q_s: projected.field_entropy.into_value() / squared,
            c_q_sigma_s: projected.field_cross.into_value() / squared,
            p_r: projected.curvature.into_value(),
            p_s: projected.entropy.into_value(),
            c_rs: projected.cross.into_value(),
            r_c: projected.correlation,
        })
    }

    /// Projects one evolved state, also returning the kinematics so that the
    /// entropy orientation carries into the next sample.
    fn project_state(
        &self,
        system: &ConformalPerturbationSystem<P, G, 2>,
        state: &ConformalPerturbationState<2>,
        settings: &SpectrumRunSettings,
        previous_normal: Option<&ContravariantField<M0, f64, 2>>,
    ) -> Result<(ProjectedSpectra, TwoFieldKinematics), SpectrumRunError> {
        let cosmic = state.cosmic();
        let kinematics = self
            .oriented_kinematics(&cosmic.background, previous_normal)
            .map_err(SpectrumRunError::KinematicBasis)?;
        let metric = self.geometry.metric(&cosmic.background.phi);
        let hubble = self.hubble(&cosmic.background);
        let matrix = field_power_spectrum(system.k, &mode_covariance(&cosmic));
        let projected = project(
            &matrix,
            &metric,
            &kinematics.basis,
            hubble,
            settings.spectrum_tolerances,
        )
        .map_err(|source| SpectrumRunError::Spectrum {
            k: system.k.into_value(),
            source,
        })?;

        Ok((projected, kinematics))
    }
}

fn component_pair(vector: &ContravariantField<M0, f64, 2>) -> [f64; 2] {
    let components = vector.components();

    [components[0].into_value(), components[1].into_value()]
}

/// The background-only tolerances implied by a run's settings.
/// Background tolerances for `N` fields. The settings carry per-field
/// tolerances for two fields; other `N` use the first field's for every field.
fn background_tolerances<const N: usize>(
    settings: &SpectrumRunSettings,
) -> ErrorTolerances<MultifieldStateDelta<N>> {
    let absolute = settings.tolerances.absolute.background;
    let index = |i: usize| if N == 2 { i } else { 0 };
    ErrorTolerances::new(
        MultifieldStateDelta {
            phi: core::array::from_fn(|i| absolute.phi[index(i)]),
            pi: core::array::from_fn(|i| absolute.pi[index(i)]),
            // The survey still uses the legacy cosmic-time background.  Its
            // scale factor is dimensionless like ln(a); the relative term
            // controls it away from zero, while this shared floor only guards
            // the initial endpoint.
            a: ScaleFactor::new(absolute.log_scale_factor.into_value()),
        },
        settings.tolerances.relative,
    )
}

/// The depth `x_init = k/(aH)` at which modes are initialized.
fn start_depth(settings: &SpectrumRunSettings) -> f64 {
    settings.e_folds_before_exit.exp()
}

/// The logarithmically spaced grid point at `index`.
fn grid_point(settings: &SpectrumRunSettings, index: usize) -> f64 {
    if settings.k_count == 1 {
        return settings.k_min_over_pivot;
    }

    let low = settings.k_min_over_pivot.ln();
    let high = settings.k_max_over_pivot.ln();
    let fraction = index as f64 / (settings.k_count - 1) as f64;

    (low + (high - low) * fraction).exp()
}

pub(crate) fn validate(settings: &SpectrumRunSettings) -> Result<(), SpectrumRunError> {
    if settings.k_count == 0 {
        return Err(SpectrumRunError::InvalidSettings(
            "k_count must be positive",
        ));
    }
    if !settings.k_min_over_pivot.is_finite()
        || !settings.k_max_over_pivot.is_finite()
        || settings.k_min_over_pivot <= 0.0
        || settings.k_max_over_pivot < settings.k_min_over_pivot
    {
        return Err(SpectrumRunError::InvalidSettings(
            "the k grid must be positive and ordered",
        ));
    }
    if !settings.e_folds_before_exit.is_finite()
        || settings.e_folds_before_exit <= 0.0
        || !start_depth(settings).is_finite()
    {
        return Err(SpectrumRunError::InvalidSettings(
            "the mode start must be finite and before horizon exit",
        ));
    }
    if !settings.e_folds_after_end.is_finite() || settings.e_folds_after_end < 0.0 {
        return Err(SpectrumRunError::InvalidSettings(
            "e_folds_after_end must be finite and non-negative",
        ));
    }
    if !settings.pivot_e_folds_before_end.is_finite()
        || settings.pivot_e_folds_before_end < 0.0
        || !settings.output_e_folds_before_end.is_finite()
        || settings.output_e_folds_before_end < 0.0
    {
        return Err(SpectrumRunError::InvalidSettings(
            "pivot and output times must be finite and no later than inflation end",
        ));
    }
    if let EndOfInflation::FieldValue { field, value } = settings.end_of_inflation
        && (field >= 2 || !value.into_value().is_finite())
    {
        return Err(SpectrumRunError::InvalidSettings(
            "a field-value end of inflation needs a field index below 2 and a finite value",
        ));
    }
    if !settings.max_e_folds.is_finite() || settings.max_e_folds <= 0.0 {
        return Err(SpectrumRunError::InvalidSettings(
            "max_e_folds must be positive and finite",
        ));
    }
    if settings.max_steps == 0 {
        return Err(SpectrumRunError::InvalidSettings(
            "max_steps must be positive",
        ));
    }

    let initial_dt = settings.initial_dt.into_value();
    let min_dt = settings.min_dt.into_value();
    let max_dt = settings.max_dt.into_value();
    if !initial_dt.is_finite()
        || !min_dt.is_finite()
        || !max_dt.is_finite()
        || min_dt <= 0.0
        || initial_dt < min_dt
        || initial_dt > max_dt
        || max_dt < min_dt
    {
        return Err(SpectrumRunError::InvalidSettings(
            "step sizes must be positive, finite, and ordered",
        ));
    }
    let absolute = &settings.tolerances.absolute;
    let valid_absolute = absolute
        .background
        .phi
        .iter()
        .all(|value| value.into_value().is_finite() && value.into_value() >= 0.0)
        && absolute
            .background
            .pi
            .iter()
            .all(|value| value.into_value().is_finite() && value.into_value() >= 0.0)
        && absolute
            .background
            .log_scale_factor
            .into_value()
            .is_finite()
        && absolute.background.log_scale_factor.into_value() >= 0.0
        && absolute.rescaled_amplitude.into_value().is_finite()
        && absolute.rescaled_amplitude.into_value() >= 0.0
        && absolute.rescaled_velocity.into_value().is_finite()
        && absolute.rescaled_velocity.into_value() >= 0.0;
    if !valid_absolute
        || !settings.tolerances.relative.is_finite()
        || settings.tolerances.relative <= 0.0
    {
        return Err(SpectrumRunError::InvalidSettings(
            "integration tolerances must be finite and non-negative, with positive relative tolerance",
        ));
    }
    let spectrum = settings.spectrum_tolerances;
    if !spectrum.hermiticity.is_finite()
        || spectrum.hermiticity < 0.0
        || !spectrum.imaginary_residual.is_finite()
        || spectrum.imaginary_residual < 0.0
        || !spectrum.correlation_overshoot.is_finite()
        || spectrum.correlation_overshoot < 0.0
    {
        return Err(SpectrumRunError::InvalidSettings(
            "spectrum tolerances must be finite and non-negative",
        ));
    }
    if settings.background_sample_every == 0 || settings.mode_sample_every == 0 {
        return Err(SpectrumRunError::InvalidSettings(
            "sampling intervals must be positive",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use super::*;
    use crate::{
        Field, FieldVelocity, FlatFieldSpace, HyperbolicFieldSpace, PerturbationState,
        PerturbationSystem, PerturbationTolerance, QuadraticMultifieldPotential, ScaleFactor,
        TwoPointBlock, TwoPointPhaseSpaceCovariance, TwoPointTransportSystem,
        TwoPointTransportTolerance, multifield::MultifieldState, rk4_step,
        two_point_covariance_from_mode_state,
    };

    type TestSystem = CovariantMultifieldSystem<QuadraticMultifieldPotential<2>, FlatFieldSpace, 2>;

    fn system(masses: [f64; 2]) -> TestSystem {
        CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(masses[0]), Mass::new(masses[1])]),
            FlatFieldSpace,
        )
    }

    fn slow_roll_start_curved<G>(
        system: &CovariantMultifieldSystem<QuadraticMultifieldPotential<2>, G, 2>,
        phi: [f64; 2],
    ) -> MultifieldState<2>
    where
        G: crate::FieldSpaceGeometry<2>,
    {
        let resting = MultifieldState::new(
            [Field::new(phi[0]), Field::new(phi[1])],
            [FieldVelocity::new(0.0); 2],
            ScaleFactor::new(1.0),
        );

        MultifieldState {
            pi: system.slow_roll_velocity(&resting),
            ..resting
        }
    }

    fn slow_roll_start(system: &TestSystem, phi: [f64; 2]) -> MultifieldState<2> {
        let resting = MultifieldState::new(
            [Field::new(phi[0]), Field::new(phi[1])],
            [FieldVelocity::new(0.0); 2],
            ScaleFactor::new(1.0),
        );

        MultifieldState {
            pi: system.slow_roll_velocity(&resting),
            ..resting
        }
    }

    /// Flat field space that declares `phi^1 < 12` outside its domain.
    #[derive(Clone, Copy, Debug)]
    struct BoundedFlat;

    impl crate::FieldSpaceGeometry<2> for BoundedFlat {
        fn metric(&self, position: &[Field; 2]) -> crate::FieldMetric<2> {
            FlatFieldSpace.metric(position)
        }

        fn inverse_metric(&self, position: &[Field; 2]) -> crate::InverseFieldMetric<2> {
            FlatFieldSpace.inverse_metric(position)
        }

        fn christoffel(&self, position: &[Field; 2]) -> crate::ChristoffelSymbols<2> {
            <FlatFieldSpace as crate::FieldSpaceGeometry<2>>::christoffel(&FlatFieldSpace, position)
        }

        fn check_domain(&self, position: &[Field; 2]) -> Result<(), crate::GeometryDomainError> {
            let phi = position[0].into_value();
            if phi < 12.0 {
                Err(crate::GeometryDomainError {
                    reason: "below the test boundary",
                    value: phi,
                })
            } else {
                Ok(())
            }
        }
    }

    impl CurvedFieldSpaceGeometry<2> for BoundedFlat {
        fn riemann(&self, _position: &[Field; 2]) -> crate::RiemannTensor<2> {
            crate::RiemannTensor::zero()
        }
    }

    #[test]
    fn the_survey_stops_where_the_geometry_leaves_its_domain() {
        let flat = system([1.0e-6, 7.0e-6]);
        let bounded = CovariantMultifieldSystem::in_planck_units(flat.potential, BoundedFlat);
        let initial = slow_roll_start(&flat, [14.0, 12.0]);
        let error = bounded
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap_err();
        match error {
            SpectrumRunError::OutsideGeometryDomain { e_folds, error } => {
                assert!(e_folds > 0.0 && error.value < 12.0, "{e_folds} {error}");
            }
            other => panic!("expected a domain error, got {other}"),
        }
    }

    /// A short run that still exercises every stage.
    fn quick_settings() -> SpectrumRunSettings {
        SpectrumRunSettings {
            e_folds_before_exit: 2.0,
            pivot_e_folds_before_end: 6.0,
            output_e_folds_before_end: 2.0,
            k_min_over_pivot: 1.0,
            k_max_over_pivot: 1.0,
            k_count: 1,
            background_sample_every: 50,
            mode_sample_every: 200,
            ..SpectrumRunSettings::default()
        }
    }

    #[test]
    fn a_field_value_event_ends_the_survey_at_the_requested_crossing() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let slow_roll = quick_settings();
        let (_, _, natural_end) = system.survey_background(initial, &slow_roll).unwrap();
        let settings = SpectrumRunSettings {
            end_of_inflation: EndOfInflation::FieldValue {
                field: 0,
                value: Field::new(6.0),
            },
            ..quick_settings()
        };
        let (_, trajectory, end) = system.survey_background(initial, &settings).unwrap();
        let final_state = trajectory.state_at(end);

        assert!(end < natural_end);
        assert!((final_state.phi[0].into_value() - 6.0).abs() < 1.0e-8);
        assert!(system.epsilon_h(&final_state) < 1.0);

        // A value the trajectory cannot reach before slow roll ends.
        let unreachable = SpectrumRunSettings {
            end_of_inflation: EndOfInflation::FieldValue {
                field: 0,
                value: Field::new(-3.0),
            },
            ..quick_settings()
        };
        assert!(matches!(
            system.survey_background(initial, &unreachable),
            Err(SpectrumRunError::SlowRollEndedFirst { .. })
        ));
        let invalid = SpectrumRunSettings {
            end_of_inflation: EndOfInflation::FieldValue {
                field: 2,
                value: Field::new(1.0),
            },
            ..quick_settings()
        };
        assert!(matches!(
            validate(&invalid),
            Err(SpectrumRunError::InvalidSettings(_))
        ));
    }

    #[test]
    fn the_grid_is_logarithmic_and_hits_both_ends() {
        let settings = SpectrumRunSettings {
            k_min_over_pivot: 0.1,
            k_max_over_pivot: 10.0,
            k_count: 3,
            ..SpectrumRunSettings::default()
        };

        assert!((grid_point(&settings, 0) - 0.1).abs() < 1.0e-12);
        assert!((grid_point(&settings, 1) - 1.0).abs() < 1.0e-12);
        assert!((grid_point(&settings, 2) - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn a_single_point_grid_uses_the_lower_end() {
        let settings = SpectrumRunSettings {
            k_count: 1,
            k_min_over_pivot: 2.0,
            ..SpectrumRunSettings::default()
        };

        assert_eq!(grid_point(&settings, 0), 2.0);
    }

    #[test]
    fn invalid_settings_are_rejected_before_any_integration() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let settings = SpectrumRunSettings {
            k_count: 0,
            ..SpectrumRunSettings::default()
        };

        assert!(matches!(
            system.run_two_field_spectra(initial, &settings),
            Err(SpectrumRunError::InvalidSettings(_))
        ));
    }

    #[test]
    fn every_numerical_setting_is_validated_before_integration() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let invalid = [
            SpectrumRunSettings {
                initial_dt: Time::new(1.0e6),
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                max_steps: 0,
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                max_e_folds: f64::NAN,
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                e_folds_before_exit: 1.0e3,
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                pivot_e_folds_before_end: -1.0,
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                output_e_folds_before_end: f64::NAN,
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                k_max_over_pivot: f64::INFINITY,
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                tolerances: ErrorTolerances::new(
                    ConformalPerturbationTolerance::background_in_planck_units(1.0e-20),
                    f64::NAN,
                ),
                ..SpectrumRunSettings::default()
            },
            SpectrumRunSettings {
                spectrum_tolerances: SpectrumTolerances {
                    imaginary_residual: -1.0,
                    ..SpectrumTolerances::default()
                },
                ..SpectrumRunSettings::default()
            },
        ];

        for settings in invalid {
            assert!(matches!(
                system.run_two_field_spectra(initial, &settings),
                Err(SpectrumRunError::InvalidSettings(_))
            ));
        }
    }

    #[test]
    fn integration_errors_report_the_stage_and_implement_error() {
        fn accepts_error(_: &dyn std::error::Error) {}

        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let settings = SpectrumRunSettings {
            max_steps: 1,
            ..SpectrumRunSettings::default()
        };
        let error = system
            .run_two_field_spectra(initial, &settings)
            .unwrap_err();

        accepts_error(&error);
        assert_eq!(
            error,
            SpectrumRunError::StepLimitReached {
                stage: SpectrumRunStage::BackgroundSurvey,
            }
        );
        assert!(error.to_string().contains("background survey"));
    }

    #[test]
    fn a_run_without_enough_e_folds_is_reported() {
        let system = system([1.0e-6, 1.5e-6]);
        // Starting below the slow-roll attractor ends inflation almost at once.
        let initial = slow_roll_start(&system, [0.4, 0.3]);
        let settings = SpectrumRunSettings {
            pivot_e_folds_before_end: 55.0,
            ..quick_settings()
        };

        assert!(matches!(
            system.run_two_field_spectra(initial, &settings),
            Err(SpectrumRunError::NotEnoughEFolds { .. })
        ));
    }

    #[test]
    fn a_short_run_produces_rows_and_a_positive_pivot() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        assert!(run.pivot.k.into_value() > 0.0);
        assert!(run.pivot.e_folds_at_end > run.pivot.e_folds_at_pivot);
        assert!(!run.background.is_empty());
        assert_eq!(run.spectra.len(), 1);
        assert_eq!(run.horizon_exit_spectra.len(), 1);
        assert!(!run.mode_evolution.is_empty());

        let spectrum = run.spectra[0];
        assert!(spectrum.p_r > 0.0);
        assert!(spectrum.p_s > 0.0);
        assert!(spectrum.r_c.abs() <= 1.0);
        assert!(spectrum.p_11 > 0.0 && spectrum.p_22 > 0.0);

        let exit = run.horizon_exit_spectra[0];
        assert!((exit.k_over_ah - 1.0).abs() < 1.0e-12);
        assert!(exit.e_folds < spectrum.e_folds);
        assert!(exit.p_r > 0.0 && exit.p_s > 0.0);
    }

    #[test]
    fn the_background_survey_ends_where_inflation_ends() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        let last = run.background.last().unwrap();
        assert!((last.epsilon_h - 1.0).abs() < 1.0e-6);
        assert!((last.e_folds - run.pivot.e_folds_at_end).abs() < 1.0e-9);
    }

    #[test]
    fn the_survey_continues_past_the_end_when_asked() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let ending = system.survey(initial, &quick_settings()).unwrap();
        let continued = system
            .survey(
                initial,
                &SpectrumRunSettings {
                    e_folds_after_end: 0.5,
                    ..quick_settings()
                },
            )
            .unwrap();

        // The end stays at the event; the trajectory runs 0.5 e-folds past it.
        assert!((continued.e_folds_at_end - ending.e_folds_at_end).abs() < 1.0e-12);
        let last = continued.trajectory.checkpoint_e_folds().last().unwrap();
        assert!((last - continued.e_folds_at_end - 0.5).abs() < 1.0e-9);
        let after = continued
            .trajectory
            .state_at(continued.e_folds_at_end + 0.25);
        assert!(system.epsilon_h(&after) > 1.0);

        let negative = system.survey(
            initial,
            &SpectrumRunSettings {
                e_folds_after_end: -1.0,
                ..quick_settings()
            },
        );
        assert!(matches!(
            negative,
            Err(SpectrumRunError::InvalidSettings(_))
        ));
    }

    #[test]
    fn interpolation_reproduces_every_background_checkpoint() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let (_, trajectory, _) = system
            .survey_background(initial, &quick_settings())
            .unwrap();

        for checkpoint in &trajectory.checkpoints {
            let interpolated = trajectory.state_at(checkpoint.e_folds);
            for index in 0..2 {
                let phi_scale = checkpoint.state.phi[index].into_value().abs().max(1.0);
                let pi_scale = checkpoint.state.pi[index].into_value().abs().max(1.0e-30);
                assert!(
                    (interpolated.phi[index] - checkpoint.state.phi[index])
                        .into_value()
                        .abs()
                        <= 1.0e-13 * phi_scale
                );
                assert!(
                    (interpolated.pi[index] - checkpoint.state.pi[index])
                        .into_value()
                        .abs()
                        <= 1.0e-13 * pi_scale
                );
            }
        }
    }

    #[test]
    fn interpolation_agrees_with_direct_background_evolution_near_the_sharpest_turn() {
        let system = system([1.0e-6, 7.0e-6]);
        let initial = slow_roll_start(&system, [14.0, 12.0]);
        let settings = SpectrumRunSettings {
            pivot_e_folds_before_end: 55.0,
            output_e_folds_before_end: 5.0,
            ..quick_settings()
        };
        let (_, trajectory, _) = system.survey_background(initial, &settings).unwrap();
        let turn_index = trajectory
            .checkpoints
            .iter()
            .take(trajectory.checkpoints.len() - 1)
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                system
                    .oriented_kinematics(&left.state, None)
                    .unwrap()
                    .eta_perp
                    .into_value()
                    .abs()
                    .total_cmp(
                        &system
                            .oriented_kinematics(&right.state, None)
                            .unwrap()
                            .eta_perp
                            .into_value()
                            .abs(),
                    )
            })
            .unwrap()
            .0;
        let target = 0.5
            * (trajectory.checkpoints[turn_index].e_folds
                + trajectory.checkpoints[turn_index + 1].e_folds);
        let (direct, _) = advance_until(
            &system,
            initial,
            Time::new(0.0),
            AdvanceControl {
                settings: &settings,
                tolerances: &background_tolerances(&settings),
                stage: SpectrumRunStage::BackgroundSurvey,
            },
            |state| target - state.e_folds(&initial),
            |_, _| {},
        )
        .unwrap();
        let interpolated = trajectory.state_at(target);

        for index in 0..2 {
            let phi_relative = ((interpolated.phi[index] - direct.phi[index]) / direct.phi[index])
                .into_value()
                .abs();
            let pi_relative = ((interpolated.pi[index] - direct.pi[index]) / direct.pi[index])
                .into_value()
                .abs();
            assert!(
                phi_relative < 1.0e-8,
                "phi relative error = {phi_relative:e}"
            );
            assert!(pi_relative < 1.0e-8, "pi relative error = {pi_relative:e}");
        }
    }

    #[test]
    fn interpolated_and_coupled_mode_drivers_agree() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let settings = quick_settings();
        let (_, trajectory, e_folds_at_end) = system.survey_background(initial, &settings).unwrap();
        let pivot_n = e_folds_at_end - settings.pivot_e_folds_before_end;
        let pivot_state = trajectory.state_at(pivot_n);
        let k = pivot_state.a * system.hubble(&pivot_state);
        let output_n = e_folds_at_end - settings.output_e_folds_before_end;
        let (optimized, optimized_exit, _) = system
            .evolve_one_mode(initial, &settings, &trajectory, k, 1.0, output_n)
            .unwrap();
        let (reference, reference_exit, _) = system
            .evolve_one_mode_reference(initial, &settings, k, 1.0, output_n)
            .unwrap();

        for (optimized, reference) in [
            (optimized.p_r, reference.p_r),
            (optimized.p_s, reference.p_s),
            (optimized_exit.p_r, reference_exit.p_r),
            (optimized_exit.p_s, reference_exit.p_s),
        ] {
            let relative = (optimized - reference).abs() / reference.abs().max(1.0e-300);
            assert!(
                relative < 1.0e-8,
                "relative spectrum difference = {relative:e}"
            );
        }
        assert!((optimized.c_rs - reference.c_rs).abs() < 1.0e-8);
        assert!((optimized_exit.c_rs - reference_exit.c_rs).abs() < 1.0e-8);
    }

    fn covariance_block_error<L, R, D: Copy>(
        transported: &TwoPointBlock<L, R, D, Complex64, 2>,
        oracle: &TwoPointBlock<L, R, D, Complex64, 2>,
    ) -> f64 {
        let scale = oracle
            .components()
            .iter()
            .flatten()
            .map(|component| component.value().norm())
            .fold(0.0, f64::max)
            .max(1.0e-300);

        transported
            .components()
            .iter()
            .flatten()
            .zip(oracle.components().iter().flatten())
            .map(|(transported, oracle)| (transported.value() - oracle.value()).norm() / scale)
            .fold(0.0, f64::max)
    }

    fn covariance_error(
        transported: &TwoPointPhaseSpaceCovariance<2>,
        oracle: &TwoPointPhaseSpaceCovariance<2>,
    ) -> f64 {
        covariance_block_error(&transported.qq, &oracle.qq)
            .max(covariance_block_error(&transported.qp, &oracle.qp))
            .max(covariance_block_error(&transported.pq, &oracle.pq))
            .max(covariance_block_error(&transported.pp, &oracle.pp))
    }

    fn direct_transport_output(
        system: &TestSystem,
        trajectory: &BackgroundTrajectory,
        k: Mass,
        output: f64,
        e_folds_before_exit: f64,
        relative_tolerance: f64,
    ) -> TwoPointPhaseSpaceCovariance<2> {
        let depth = e_folds_before_exit.exp();
        let start = trajectory
            .crossing(|state| (k / (state.a * system.hubble(state))).into_value() - depth)
            .unwrap();
        let start_background = trajectory.state_at(start);
        let initialized = ConformalPerturbationSystem::new(*system, k)
            .bunch_davies_state(start_background, depth)
            .unwrap();
        let transport = TwoPointTransportSystem::new(system, trajectory, k);
        let tolerances = ErrorTolerances::new(
            TwoPointTransportTolerance::pure_relative(),
            relative_tolerance,
        );
        let limits = StepSizeLimits {
            min: crate::Dimensionless::new(1.0e-8),
            max: crate::Dimensionless::new(5.0e-2),
        };
        let mut covariance = two_point_covariance_from_mode_state(system, &initialized.cosmic());
        let mut e_folds = start;
        let mut step: crate::Dimensionless = crate::Dimensionless::new(1.0e-3);

        for _ in 0..1_000_000 {
            let remaining = output - e_folds;
            if remaining <= 1.0e-13 {
                return covariance;
            }
            let attempted = crate::Dimensionless::new(step.into_value().min(remaining));
            let outcome = adaptive_step(
                &transport,
                crate::Dimensionless::new(e_folds),
                &covariance,
                attempted,
                &tolerances,
            );
            if outcome.accepted {
                covariance = outcome.state;
                e_folds += attempted.into_value();
            }
            step = next_step_size(attempted, outcome.error_ratio, &limits);
        }

        panic!("direct two-point transport exceeded its adaptive step limit")
    }

    #[test]
    fn direct_two_point_transport_matches_every_mode_covariance_block() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let settings = quick_settings();
        let (_, trajectory, e_folds_at_end) = system.survey_background(initial, &settings).unwrap();
        let horizon = e_folds_at_end - settings.pivot_e_folds_before_end;
        let output = e_folds_at_end - settings.output_e_folds_before_end;
        let pivot_state = trajectory.state_at(horizon);
        let k = pivot_state.a * system.hubble(&pivot_state);
        let start = trajectory
            .crossing(|state| {
                (k / (state.a * system.hubble(state))).into_value() - start_depth(&settings)
            })
            .unwrap();
        let turn = trajectory
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.e_folds >= start && checkpoint.e_folds <= output)
            .max_by(|left, right| {
                let left_turn = system
                    .oriented_kinematics(&left.state, None)
                    .unwrap()
                    .eta_perp
                    .into_value()
                    .abs();
                let right_turn = system
                    .oriented_kinematics(&right.state, None)
                    .unwrap()
                    .eta_perp
                    .into_value()
                    .abs();
                left_turn.total_cmp(&right_turn)
            })
            .unwrap()
            .e_folds;
        let conformal_system = ConformalPerturbationSystem::new(system, k);
        let start_background = trajectory.state_at(start);
        let initialized = conformal_system
            .bunch_davies_state(start_background, start_depth(&settings))
            .unwrap();
        let mode_system = InterpolatedModeSystem {
            background: &system,
            trajectory: &trajectory,
            k,
            tensor: false,
        };
        let transport_system = TwoPointTransportSystem::new(&system, &trajectory, k);
        let mut modes = ModesInEFolds(initialized.modes);
        let mut covariance = two_point_covariance_from_mode_state(&system, &initialized.cosmic());
        let mut e_folds = start;
        let mut targets = vec![start, horizon, turn, output];
        targets.sort_by(f64::total_cmp);
        targets.dedup_by(|left, right| (*left - *right).abs() < 1.0e-12);

        for target in targets {
            while target - e_folds > 1.0e-13 {
                let step = (target - e_folds).min(2.0e-3);
                let dn = crate::Dimensionless::new(step);
                modes = rk4_step(&mode_system, crate::Dimensionless::new(e_folds), &modes, dn);
                covariance = rk4_step(
                    &transport_system,
                    crate::Dimensionless::new(e_folds),
                    &covariance,
                    dn,
                );
                e_folds += step;
            }

            let combined = ConformalPerturbationState {
                background: ConformalBackgroundState::from_cosmic(trajectory.state_at(target)),
                modes: modes.0,
            };
            let oracle = two_point_covariance_from_mode_state(&system, &combined.cosmic());
            let error = covariance_error(&covariance, &oracle);

            assert!(error < 2.0e-7, "N={target:e}, all-block error={error:e}");
        }
    }

    #[test]
    fn direct_two_point_transport_converges_with_tighter_tolerance() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let settings = quick_settings();
        let (_, trajectory, e_folds_at_end) = system.survey_background(initial, &settings).unwrap();
        let horizon = e_folds_at_end - settings.pivot_e_folds_before_end;
        let output = e_folds_at_end - settings.output_e_folds_before_end;
        let pivot_state = trajectory.state_at(horizon);
        let k = pivot_state.a * system.hubble(&pivot_state);

        let loose = direct_transport_output(&system, &trajectory, k, output, 3.0, 1.0e-7);
        let tight = direct_transport_output(&system, &trajectory, k, output, 3.0, 1.0e-10);

        assert!(covariance_error(&loose, &tight) < 2.0e-6);
    }

    #[test]
    fn direct_two_point_transport_converges_from_deeper_bunch_davies_starts() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let settings = quick_settings();
        let (_, trajectory, e_folds_at_end) = system.survey_background(initial, &settings).unwrap();
        let horizon = e_folds_at_end - settings.pivot_e_folds_before_end;
        let output = e_folds_at_end - settings.output_e_folds_before_end;
        let pivot_state = trajectory.state_at(horizon);
        let k = pivot_state.a * system.hubble(&pivot_state);
        let run = |depth| direct_transport_output(&system, &trajectory, k, output, depth, 1.0e-10);

        let coarse = run(3.0);
        let medium = run(4.0);
        let fine = run(5.0);
        let first = covariance_error(&coarse, &medium);
        let second = covariance_error(&medium, &fine);

        assert!(second < 0.5 * first, "{first:e} then {second:e}");
    }

    #[test]
    fn a_separable_straight_trajectory_has_no_turn_and_a_small_cross_spectrum() {
        // Equal masses make the trajectory a straight line in field space, so
        // the entropy direction never sources curvature.
        let system = system([1.0e-6, 1.0e-6]);
        let initial = slow_roll_start(&system, [10.0, 10.0]);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        for row in &run.background {
            assert!(row.eta_perp.abs() < 1.0e-8, "eta_perp = {}", row.eta_perp);
        }

        let spectrum = run.spectra[0];
        let scale = spectrum.p_q_sigma.max(spectrum.p_q_s);
        assert!(
            spectrum.c_q_sigma_s.abs() < 1.0e-6 * scale,
            "cross = {}",
            spectrum.c_q_sigma_s
        );
    }

    #[test]
    fn the_single_field_limit_reproduces_the_analytic_curvature_spectrum() {
        // A trajectory confined to the first axis has no turn, so the
        // curvature spectrum must match the single-field slow-roll result
        // P_R = H_*^2 / (8 pi^2 epsilon_* M_pl^2) at horizon crossing.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 0.0]);
        let run = system
            .run_two_field_spectra(
                initial,
                &SpectrumRunSettings {
                    pivot_e_folds_before_end: 25.0,
                    ..quick_settings()
                },
            )
            .unwrap();

        let spectrum = run.spectra[0];
        assert!((spectrum.k_over_pivot - 1.0).abs() < 1.0e-12);

        let hubble = run.pivot.hubble.into_value();
        let expected = hubble * hubble / (8.0 * PI * PI * run.pivot.epsilon_h);
        let relative = (spectrum.p_r - expected).abs() / expected;

        // The formula is leading order in slow roll, so its own error is
        // O(epsilon_*). A fixed threshold would either be vacuous or fail
        // whenever the pivot moves.
        let allowed = 3.0 * run.pivot.epsilon_h;
        assert!(
            relative < allowed,
            "P_R = {}, expected {}, relative error {} exceeds {}",
            spectrum.p_r,
            expected,
            relative,
            allowed
        );
    }

    #[test]
    fn the_tensor_spectrum_matches_slow_roll_and_the_consistency_relation() {
        // On a trajectory along the first axis the model is single-field
        // quadratic inflation. At second order in slow roll (Stewart and
        // Lyth), P_t = (2 H_*^2 / pi^2)[1 - 2 (C + 1) epsilon_1] and
        // r = 16 epsilon_1 [1 + C epsilon_2], with C = -2 + ln 2 + gamma_E and
        // epsilon_2 = d ln epsilon_1 / dN, all at horizon crossing.
        let system = system([1.0e-6, 1.5e-6]);
        let c = -2.0 + 2.0_f64.ln() + 0.577_215_664_901_532_9;
        for (phi, pivot_before_end) in [(12.0, 25.0), (15.0, 45.0)] {
            let initial = slow_roll_start(&system, [phi, 0.0]);
            let settings = SpectrumRunSettings {
                pivot_e_folds_before_end: pivot_before_end,
                e_folds_before_exit: 6.0,
                ..quick_settings()
            };
            let survey = system.survey(initial, &settings).unwrap();
            let output = survey.e_folds_at_end - settings.output_e_folds_before_end;
            let crossing = survey.e_folds_at_end - settings.pivot_e_folds_before_end;
            let pivot = survey.trajectory.state_at(crossing);
            let k = pivot.a * system.hubble(&pivot);
            let hubble = system.hubble(&pivot).into_value();
            let epsilon = system.epsilon_h(&pivot);
            let step = 0.01;
            let epsilon_at = |n: f64| system.epsilon_h(&survey.trajectory.state_at(n));
            let epsilon_2 = (epsilon_at(crossing + step).ln() - epsilon_at(crossing - step).ln())
                / (2.0 * step);
            let tensor = system
                .tensor_spectrum(&settings, &survey.trajectory, k, output)
                .unwrap();
            let scalar = system
                .mode_spectrum(&settings, &survey.trajectory, k, output)
                .unwrap()
                .spectra
                .curvature
                .into_value();
            let slow_roll = 2.0 * hubble * hubble / (PI * PI) * (1.0 - 2.0 * (c + 1.0) * epsilon);
            let consistency = 16.0 * epsilon * (1.0 + c * epsilon_2);
            let r = tensor / scalar;
            assert!(
                (tensor / slow_roll - 1.0).abs() < 2.0 * epsilon * epsilon,
                "P_t {tensor} vs {slow_roll} at epsilon {epsilon}"
            );
            assert!(
                (r / consistency - 1.0).abs() < 2.0 * epsilon * epsilon,
                "r {r} vs {consistency} at epsilon {epsilon}"
            );
        }
    }

    #[test]
    fn the_covariance_stays_positive_semidefinite() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        let spectrum = run.spectra[0];
        let determinant = spectrum.p_11 * spectrum.p_22 - spectrum.re_p_12 * spectrum.re_p_12;

        assert!(spectrum.p_11 > 0.0);
        assert!(spectrum.p_11 + spectrum.p_22 > 0.0);
        assert!(
            determinant > -1.0e-12 * spectrum.p_11 * spectrum.p_22,
            "determinant = {determinant}"
        );
    }

    #[test]
    fn the_spectrum_does_not_depend_on_where_the_modes_start() {
        // The Bunch-Davies form is approached as 1/x_init^2, so the two
        // depths the design recommends must agree far more closely than
        // either agrees with a shallow start.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);

        let shallow = system
            .run_two_field_spectra(
                initial,
                &SpectrumRunSettings {
                    e_folds_before_exit: 4.0,
                    ..quick_settings()
                },
            )
            .unwrap();
        let deep = system
            .run_two_field_spectra(
                initial,
                &SpectrumRunSettings {
                    e_folds_before_exit: 5.0,
                    ..quick_settings()
                },
            )
            .unwrap();

        let relative = (shallow.spectra[0].p_r - deep.spectra[0].p_r).abs() / deep.spectra[0].p_r;

        assert!(relative < 1.0e-3, "relative difference {relative}");
    }

    #[test]
    fn starting_deeper_inside_the_horizon_converges() {
        // The Bunch-Davies form used here is leading order, so its error
        // falls off as 1/x_init^2. Successive doublings must therefore move
        // the answer by ever less. A flat sequence would mean the mode
        // components are not actually being controlled to the requested
        // accuracy, which is exactly what a mis-scaled absolute tolerance
        // produces.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let spectrum = |e_folds_before_exit| {
            system
                .run_two_field_spectra(
                    initial,
                    &SpectrumRunSettings {
                        e_folds_before_exit,
                        ..quick_settings()
                    },
                )
                .unwrap()
                .spectra[0]
                .p_r
        };

        let coarse = spectrum(4.0);
        let medium = spectrum(5.0);
        let fine = spectrum(6.0);
        let finest = spectrum(7.0);

        let first = (medium - coarse).abs() / medium;
        let second = (fine - medium).abs() / fine;
        let third = (finest - fine).abs() / finest;

        assert!(second < 0.5 * first, "{first} then {second}");
        assert!(third < 0.5 * second, "{second} then {third}");
    }

    #[test]
    fn the_answer_is_insensitive_to_the_absolute_tolerance() {
        // The background floor and the mode components are now separate
        // settings, so moving the background floor over thirty orders of
        // magnitude cannot reach the modes at all.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let curvature = |absolute| {
            system
                .run_two_field_spectra(
                    initial,
                    &SpectrumRunSettings {
                        tolerances: ErrorTolerances::new(
                            ConformalPerturbationTolerance::background_in_planck_units(absolute),
                            1.0e-10,
                        ),
                        ..quick_settings()
                    },
                )
                .unwrap()
                .spectra[0]
                .p_r
        };

        let reference = curvature(1.0e-45);
        for absolute in [1.0e-30, 1.0e-20, 1.0e-16] {
            let value = curvature(absolute);
            let relative = (value - reference).abs() / reference;

            // Raising the floor across twenty-nine orders of magnitude moves
            // the answer by less than a part in ten million.
            assert!(
                relative < 1.0e-7,
                "absolute {absolute:e} moved P_R by {relative:e}"
            );
        }
    }

    #[test]
    fn the_default_leaves_the_modes_under_pure_relative_control() {
        // A mode error is normalized by a modulus, which does not vanish for
        // a physical mode, so no floor is needed to keep the ratio finite.
        // A floor here would only cap the accuracy of the modes.
        let absolute = SpectrumRunSettings::default().tolerances.absolute;

        assert_eq!(absolute.rescaled_amplitude.into_value(), 0.0);
        assert_eq!(absolute.rescaled_velocity.into_value(), 0.0);
        assert!(absolute.background.phi[0].into_value() > 0.0);
    }

    /// Advances a mode state to a requested e-folding number.
    fn advance_modes<P, G>(
        system: &PerturbationSystem<P, G, 2>,
        settings: &SpectrumRunSettings,
        initial: &MultifieldState<2>,
        state: PerturbationState<2>,
        time: Time,
        target: f64,
    ) -> (PerturbationState<2>, Time)
    where
        P: TwiceDifferentiableMultifieldPotential<2> + Clone,
        G: CurvedFieldSpaceGeometry<2> + Clone,
    {
        let absolute = settings.tolerances.absolute;
        let cosmic_tolerances = ErrorTolerances::new(
            PerturbationTolerance {
                background: MultifieldStateDelta {
                    phi: absolute.background.phi,
                    pi: absolute.background.pi,
                    a: ScaleFactor::new(absolute.background.log_scale_factor.into_value()),
                },
                rescaled_amplitude: absolute.rescaled_amplitude,
                rescaled_velocity: absolute.rescaled_velocity,
            },
            settings.tolerances.relative,
        );
        advance_until(
            system,
            state,
            time,
            AdvanceControl {
                settings,
                tolerances: &cosmic_tolerances,
                stage: SpectrumRunStage::ModeEvolution {
                    k: system.k.into_value(),
                },
            },
            |state| target - state.background.e_folds(initial),
            |_, _| {},
        )
        .unwrap()
    }

    /// `[[R_1, R_2], [S_1, S_2]]` for one evolved state.
    fn projections<P, G>(
        system: &PerturbationSystem<P, G, 2>,
        state: &PerturbationState<2>,
    ) -> crate::ModeProjections
    where
        P: TwiceDifferentiableMultifieldPotential<2> + Clone,
        G: CurvedFieldSpaceGeometry<2> + Clone,
    {
        system.mode_projections(state, None).unwrap()
    }

    #[test]
    fn two_light_fields_freeze_at_the_de_sitter_amplitude() {
        // The `(H_*/2 pi)^2` result needs a window: earlier than about two
        // e-folds after exit the exact mode still carries its `1 + x^2`
        // correction, and later the entropy field decays under its own mass
        // while the adiabatic one grows with `2 epsilon`. Both effects are
        // measured in the next test; here the window is used.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let run = system
            .run_two_field_spectra(
                initial,
                &SpectrumRunSettings {
                    pivot_e_folds_before_end: 30.0,
                    output_e_folds_before_end: 28.0,
                    ..quick_settings()
                },
            )
            .unwrap();

        let hubble = run.pivot.hubble.into_value();
        let de_sitter = (hubble / (2.0 * PI)).powi(2);
        let spectrum = run.spectra[0];

        for (name, value) in [("P_Qsigma", spectrum.p_q_sigma), ("P_Qs", spectrum.p_q_s)] {
            let ratio = value / de_sitter;
            assert!(
                (ratio - 1.0).abs() < 0.15,
                "{name} / (H/2pi)^2 = {ratio}, heaviest m^2/H^2 = {}",
                (1.5e-6 / hubble).powi(2)
            );
        }
    }

    #[test]
    fn super_horizon_drift_follows_the_analytic_rates() {
        // Between two and eight e-folds after exit the entropy spectrum decays
        // as a light field of mass m does, and the adiabatic field spectrum
        // grows with 2 epsilon because Q_sigma = (dot sigma / H) R and R is
        // conserved.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let at = |after: f64| {
            system
                .run_two_field_spectra(
                    initial,
                    &SpectrumRunSettings {
                        pivot_e_folds_before_end: 30.0,
                        output_e_folds_before_end: 30.0 - after,
                        ..quick_settings()
                    },
                )
                .unwrap()
        };

        let early = at(2.0);
        let late = at(8.0);
        let elapsed = 6.0;

        // A light entropy field decays as P_S ~ a^(-2 m^2 / 3 H^2).
        let mass_ratio = (1.5e-6 / early.pivot.hubble.into_value()).powi(2);
        let predicted_decay = (-2.0 * mass_ratio * elapsed / 3.0).exp();
        let measured_decay = late.spectra[0].p_q_s / early.spectra[0].p_q_s;
        assert!(
            (measured_decay / predicted_decay - 1.0).abs() < 0.1,
            "entropy decayed by {measured_decay}, predicted {predicted_decay}"
        );

        // The adiabatic field spectrum tracks 2 epsilon, which for a quadratic
        // model is 1/N e-folds before the end.
        let predicted_growth = (30.0 - 2.0) / (30.0 - 8.0);
        let measured_growth = late.spectra[0].p_q_sigma / early.spectra[0].p_q_sigma;
        assert!(
            (measured_growth / predicted_growth - 1.0).abs() < 0.1,
            "adiabatic grew by {measured_growth}, predicted {predicted_growth}"
        );
    }

    #[test]
    fn the_super_horizon_transfer_matrix_is_triangular() {
        // Wands' statement: entropy can source curvature, but a purely
        // adiabatic perturbation creates no entropy. Two independent
        // solutions determine the transfer matrix as T = M_late M_early^-1,
        // whose entropy-from-curvature entry must vanish.
        let background = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&background, [12.0, 8.0]);
        let settings = quick_settings();
        let perturbation = PerturbationSystem::new(background, Mass::new(6.0e-5));

        let (start, start_time) = advance_until(
            &background,
            initial,
            Time::new(0.0),
            AdvanceControl {
                settings: &settings,
                tolerances: &background_tolerances(&settings),
                stage: SpectrumRunStage::ModeStart {
                    k: perturbation.k.into_value(),
                },
            },
            |state| {
                (perturbation.k / (state.a * background.hubble(state))).into_value()
                    - start_depth(&settings)
            },
            |_, _| {},
        )
        .unwrap();
        let exit = start.e_folds(&initial) + settings.e_folds_before_exit;
        let mode_state = perturbation
            .bunch_davies_state(start, start_depth(&settings))
            .unwrap();

        let (early, early_time) = advance_modes(
            &perturbation,
            &settings,
            &initial,
            mode_state,
            start_time,
            exit + 2.0,
        );
        let (late, _) = advance_modes(
            &perturbation,
            &settings,
            &initial,
            early,
            early_time,
            exit + 6.0,
        );

        let m_early = projections(&perturbation, &early);
        let m_late = projections(&perturbation, &late);
        let determinant = m_early[0][0] * m_early[1][1] - m_early[0][1] * m_early[1][0];
        let inverse = [
            [m_early[1][1] / determinant, -m_early[0][1] / determinant],
            [-m_early[1][0] / determinant, m_early[0][0] / determinant],
        ];
        let transfer: [[_; 2]; 2] = core::array::from_fn(|row| {
            core::array::from_fn(|column| {
                m_late[row][0] * inverse[0][column] + m_late[row][1] * inverse[1][column]
            })
        });

        let entropy_from_curvature = transfer[1][0].norm();
        let entropy_from_entropy = transfer[1][1].norm();
        let curvature_from_entropy = transfer[0][1].norm();

        // The forbidden entry is not exactly zero, because the early time is
        // two e-folds outside the horizon rather than infinitely far, leaving
        // gradient corrections of order `(k/aH)^2 ~= 0.018`. It comes out far
        // below even that.
        assert!(
            entropy_from_curvature < 1.0e-3 * entropy_from_entropy,
            "T_SR = {entropy_from_curvature}, T_SS = {entropy_from_entropy}"
        );
        // The other off-diagonal entry is the physical sourcing, which a
        // turning trajectory must actually produce, so the triangle has a
        // direction rather than simply being small everywhere.
        assert!(
            curvature_from_entropy > 100.0 * entropy_from_curvature,
            "T_RS = {curvature_from_entropy}, T_SR = {entropy_from_curvature}"
        );
    }

    #[test]
    fn tighter_tolerances_do_not_move_the_answer() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let curvature = |relative| {
            system
                .run_two_field_spectra(
                    initial,
                    &SpectrumRunSettings {
                        tolerances: ErrorTolerances::new(
                            ConformalPerturbationTolerance::background_in_planck_units(1.0e-20),
                            relative,
                        ),
                        ..quick_settings()
                    },
                )
                .unwrap()
                .spectra[0]
                .p_r
        };

        let loose = curvature(1.0e-9);
        let tight = curvature(1.0e-12);

        assert!((loose - tight).abs() / tight < 1.0e-7);
    }

    #[test]
    fn a_denser_grid_does_not_disturb_the_wavenumbers_it_shares() {
        // Each wavenumber is integrated from the same initial state, so a
        // denser grid must reproduce the coarser one exactly. Anything else
        // would mean state leaking between grid points.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let grid = |count| {
            system
                .run_two_field_spectra(
                    initial,
                    &SpectrumRunSettings {
                        k_min_over_pivot: 0.5,
                        k_max_over_pivot: 2.0,
                        k_count: count,
                        ..quick_settings()
                    },
                )
                .unwrap()
                .spectra
        };

        let coarse = grid(3);
        let dense = grid(5);

        assert_eq!(coarse[1].k_over_pivot, dense[2].k_over_pivot);
        assert_eq!(coarse[1].p_r, dense[2].p_r);
        assert_eq!(coarse[1].p_s, dense[2].p_s);
    }

    #[test]
    fn probe_nearly_aligned_trajectory() {
        let system = system([1.0e-6, 1.5e-6]);
        for second in [0.0, 1.0e-8, 1.0e-4, 1.0e-2, 1.0] {
            let initial = slow_roll_start(&system, [12.0, second]);
            let outcome = system.run_two_field_spectra(initial, &quick_settings());
            match outcome {
                Ok(run) => println!("phi_2 = {second:e}: ok, P_R = {:.6e}", run.spectra[0].p_r),
                Err(error) => println!("phi_2 = {second:e}: {error:?}"),
            }
        }
    }

    #[test]
    fn a_mode_still_inside_the_horizon_is_refused() {
        // With the pivot exiting six e-folds before the end, asking for
        // output earlier than that leaves the mode inside the horizon, where
        // its "spectrum" is just the vacuum it was initialized with.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let at = |output| {
            system.run_two_field_spectra(
                initial,
                &SpectrumRunSettings {
                    output_e_folds_before_end: output,
                    ..quick_settings()
                },
            )
        };

        assert!(at(2.0).is_ok());
        for output in [6.5, 7.5] {
            match at(output) {
                Err(SpectrumRunError::OutputBeforeHorizonExit { k_over_ah, .. }) => {
                    assert!(k_over_ah >= 1.0, "reported k/(aH) = {k_over_ah}");
                }
                other => panic!("expected a refusal at output {output}, got {other:?}"),
            }
        }
    }

    /// A curvature radius and start that leave inflation ending within the
    /// e-fold budget while making the Riemann term large enough to measure.
    /// Stronger curvature with a larger second field sustains inflation past
    /// the budget, which is hyperinflation rather than a failure.
    const CURVATURE_RADIUS: f64 = 5.0;
    const CURVED_START: [f64; 2] = [12.0, 8.0];

    fn curved_system(
        radius: f64,
    ) -> CovariantMultifieldSystem<QuadraticMultifieldPotential<2>, HyperbolicFieldSpace, 2> {
        CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(1.0e-6), Mass::new(1.5e-6)]),
            HyperbolicFieldSpace::new(Field::new(radius)).unwrap(),
        )
    }

    /// The hyperbolic geometry with its Riemann tensor forced to zero.
    ///
    /// This is deliberately not a consistent geometry: the metric it reports
    /// does have curvature. It exists so that a test can hold the metric, the
    /// inverse metric and the connection fixed and vary only the curvature
    /// contribution, which is otherwise entangled with the Christoffel
    /// correction to the potential Hessian.
    #[derive(Clone, Copy)]
    struct RiemannFree(HyperbolicFieldSpace);

    impl crate::FieldSpaceGeometry<2> for RiemannFree {
        fn metric(&self, position: &[Field; 2]) -> crate::FieldMetric<2> {
            self.0.metric(position)
        }

        fn inverse_metric(&self, position: &[Field; 2]) -> crate::InverseFieldMetric<2> {
            self.0.inverse_metric(position)
        }

        fn christoffel(&self, position: &[Field; 2]) -> crate::ChristoffelSymbols<2> {
            self.0.christoffel(position)
        }
    }

    impl CurvedFieldSpaceGeometry<2> for RiemannFree {
        fn riemann(&self, _position: &[Field; 2]) -> crate::RiemannTensor<2> {
            crate::RiemannTensor::new([[[[crate::Quantity::new(0.0); 2]; 2]; 2]; 2])
        }
    }

    fn riemann_free_system(
        radius: f64,
    ) -> CovariantMultifieldSystem<QuadraticMultifieldPotential<2>, RiemannFree, 2> {
        CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(1.0e-6), Mass::new(1.5e-6)]),
            RiemannFree(HyperbolicFieldSpace::new(Field::new(radius)).unwrap()),
        )
    }

    #[test]
    fn field_space_curvature_reaches_the_perturbation_mass_matrix() {
        // Every other test in the crate runs on a flat field space, where the
        // Riemann contribution to the mass matrix is identically zero. Holding
        // the metric and connection fixed and zeroing only the curvature
        // isolates that contribution.
        let system = curved_system(CURVATURE_RADIUS);
        let state = slow_roll_start_curved(&system, CURVED_START);

        let curved = system.perturbation_mass_matrix(&state);
        let without = riemann_free_system(CURVATURE_RADIUS).perturbation_mass_matrix(&state);

        let entries = |matrix: &crate::FieldTensor2<_, _, _, f64, 2>| {
            (0..2)
                .flat_map(|row| (0..2).map(move |column| (row, column)))
                .map(|(row, column)| matrix.components()[row][column].into_value())
                .collect::<Vec<_>>()
        };
        let difference = entries(&curved)
            .iter()
            .zip(entries(&without))
            .map(|(with, out)| (with - out).abs())
            .fold(0.0, f64::max);
        let scale = entries(&without)
            .iter()
            .map(|entry| entry.abs())
            .fold(0.0, f64::max);

        // Measured at 0.39% of the largest entry. The Riemann term carries
        // two factors of the field velocity, so it is smallest at the start of
        // a slow-roll trajectory and grows with epsilon; the spectra it feeds
        // move by rather more.
        assert!(
            difference > 1.0e-3 * scale,
            "curvature moved the mass matrix by {difference} against a scale of {scale}"
        );
    }

    #[test]
    fn a_curved_field_space_runs_end_to_end() {
        let system = curved_system(CURVATURE_RADIUS);
        let initial = slow_roll_start_curved(&system, CURVED_START);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        let spectrum = run.spectra[0];
        assert!(spectrum.p_r > 0.0);
        assert!(spectrum.p_s > 0.0);
        assert!(spectrum.r_c.abs() <= 1.0);
        assert!(run.pivot.k.into_value() > 0.0);
        assert!(!run.background.is_empty());
    }

    #[test]
    fn curvature_changes_the_spectrum() {
        // Same metric, same connection, same initial state; only the Riemann
        // term differs. If the spectra agree, the curvature never reached the
        // modes.
        let curved = curved_system(CURVATURE_RADIUS);
        let without = riemann_free_system(CURVATURE_RADIUS);
        let initial = slow_roll_start_curved(&curved, CURVED_START);

        let with_curvature = curved
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap()
            .spectra[0]
            .p_r;
        let without_curvature = without
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap()
            .spectra[0]
            .p_r;

        let relative = (with_curvature - without_curvature).abs() / without_curvature;

        // Measured at 6.3e-3, two orders above the run-to-run noise.
        assert!(
            relative > 1.0e-3,
            "curvature moved P_R by only {relative}: {with_curvature} against {without_curvature}"
        );
    }

    #[test]
    fn the_cross_spectrum_reports_its_imaginary_part() {
        // Twice Im P^12 forms the antisymmetric equal-time correlator
        // proportional to the field commutator. It is zero in the vacuum and
        // conserved, so an emitted residual diagnoses integration error.
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        let spectrum = run.spectra[0];
        let scale = (spectrum.p_11 * spectrum.p_22).sqrt();

        assert!(spectrum.re_p_12.abs() > 0.1 * scale);
        assert!(
            spectrum.im_p_12.abs() < 1.0e-6 * scale,
            "Im P^12 = {}, scale {scale}",
            spectrum.im_p_12
        );
    }

    #[test]
    fn csv_rows_match_their_headers() {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        let run = system
            .run_two_field_spectra(initial, &quick_settings())
            .unwrap();

        let mut buffer = Vec::new();
        run.write_spectra_csv(&mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let mut lines = text.lines();

        let header = lines.next().unwrap();
        let columns = header.split(',').count();
        assert_eq!(header, SpectrumRecord::CSV_HEADER);
        for line in lines {
            assert_eq!(line.split(',').count(), columns);
        }

        let mut horizon_buffer = Vec::new();
        run.write_horizon_exit_spectra_csv(&mut horizon_buffer)
            .unwrap();
        let horizon_text = String::from_utf8(horizon_buffer).unwrap();
        assert_eq!(
            horizon_text.lines().count(),
            run.horizon_exit_spectra.len() + 1
        );
    }

    #[test]
    fn metadata_records_the_pivot_and_tolerances() {
        let settings = quick_settings();
        let pivot = Pivot {
            k: Mass::new(1.5e-5),
            hubble: crate::Hubble::new(2.0e-6),
            epsilon_h: 0.01,
            e_folds_at_end: 12.0,
            e_folds_at_pivot: 6.0,
        };

        let mut buffer = Vec::new();
        write_metadata(&mut buffer, &settings, &pivot, Some("abc123")).unwrap();
        let text = String::from_utf8(buffer).unwrap();

        assert!(text.contains("# revision: abc123"));
        assert!(text.contains("# background_strategy: shared_cubic_hermite_in_e_folds"));
        assert!(text.contains("# e_folds_before_exit: 2e0"));
        assert!(text.contains("# k_pivot_internal_over_m_pl: 1.5e-5"));
        assert!(text.lines().all(|line| line.starts_with('#')));
    }

    fn three_point_block_error<A, B, C, D: Copy>(
        actual: &crate::ThreePointBlock<A, B, C, D, Complex64, 2>,
        reference: &crate::ThreePointBlock<A, B, C, D, Complex64, 2>,
    ) -> f64 {
        let flatten = |block: &crate::ThreePointBlock<A, B, C, D, Complex64, 2>| {
            block
                .components()
                .iter()
                .flatten()
                .flatten()
                .map(|component| *component.value())
                .collect::<Vec<Complex64>>()
        };
        let reference = flatten(reference);
        let scale = reference
            .iter()
            .map(|component| component.norm())
            .fold(0.0, f64::max)
            .max(1.0e-300);

        flatten(actual)
            .into_iter()
            .zip(reference)
            .map(|(actual, reference)| (actual - reference).norm() / scale)
            .fold(0.0, f64::max)
    }

    /// Largest block-relative difference over the eight three-point blocks.
    fn three_point_error<P: crate::PhaseSpaceSlot>(
        actual: &crate::ThreePointPhaseSpaceCorrelation<2, P>,
        reference: &crate::ThreePointPhaseSpaceCorrelation<2, P>,
    ) -> [f64; 8] {
        [
            three_point_block_error(&actual.qqq, &reference.qqq),
            three_point_block_error(&actual.qqp, &reference.qqp),
            three_point_block_error(&actual.qpq, &reference.qpq),
            three_point_block_error(&actual.pqq, &reference.pqq),
            three_point_block_error(&actual.qpp, &reference.qpp),
            three_point_block_error(&actual.pqp, &reference.pqp),
            three_point_block_error(&actual.ppq, &reference.ppq),
            three_point_block_error(&actual.ppp, &reference.ppp),
        ]
    }

    struct SourcedBenchmark<P = QuadraticMultifieldPotential<2>, G = FlatFieldSpace> {
        system: CovariantMultifieldSystem<P, G, 2>,
        trajectory: BackgroundTrajectory,
        triangle: crate::MomentumTriangle,
        output: f64,
    }

    fn sourced_benchmark(shape: [f64; 3]) -> SourcedBenchmark {
        let system = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&system, [12.0, 8.0]);
        sourced_benchmark_with(system, initial, shape, None, &quick_settings())
    }

    /// Surveys `system` from `initial`. The triangle is `shape` times the
    /// pivot wavenumber, or times `pivot` when one is supplied.
    fn sourced_benchmark_with<P, G>(
        system: CovariantMultifieldSystem<P, G, 2>,
        initial: MultifieldState<2>,
        shape: [f64; 3],
        pivot: Option<f64>,
        settings: &SpectrumRunSettings,
    ) -> SourcedBenchmark<P, G>
    where
        P: crate::ThriceDifferentiableMultifieldPotential<2> + Copy,
        G: crate::ThirdOrderFieldSpaceGeometry<2> + crate::CurvatureDerivativeGeometry<2> + Copy,
    {
        let (_, trajectory, e_folds_at_end) = system.survey_background(initial, settings).unwrap();
        let pivot_state = trajectory.state_at(e_folds_at_end - settings.pivot_e_folds_before_end);
        let k = pivot.unwrap_or((pivot_state.a * system.hubble(&pivot_state)).into_value());
        let triangle =
            crate::MomentumTriangle::try_new(shape[0] * k, shape[1] * k, shape[2] * k).unwrap();

        SourcedBenchmark {
            system,
            trajectory,
            triangle,
            output: e_folds_at_end - settings.output_e_folds_before_end,
        }
    }

    impl<P, G> SourcedBenchmark<P, G>
    where
        P: crate::ThriceDifferentiableMultifieldPotential<2> + Copy,
        G: crate::ThirdOrderFieldSpaceGeometry<2> + crate::CurvatureDerivativeGeometry<2> + Copy,
    {
        /// Elapsed e-fold at which the shortest leg has `k/(aH) = depth`.
        fn slice(&self, depth: f64) -> f64 {
            let shortest = Mass::new(self.triangle.canonical_wavenumbers()[2]);
            self.trajectory
                .crossing(|state| {
                    (shortest / (state.a * self.system.hubble(state))).into_value() - depth
                })
                .unwrap()
        }

        fn evolve(
            &self,
            e_folds_before_exit: f64,
            target: f64,
            relative_tolerance: f64,
        ) -> crate::CanonicalThreePointTransportState<2> {
            crate::bispectrum_run::evolve_three_point(
                &self.system,
                &self.trajectory,
                self.triangle,
                e_folds_before_exit,
                target,
                relative_tolerance,
                2_000_000,
            )
            .unwrap()
            .1
        }
    }

    /// Largest error over blocks with the given number of momentum slots.
    fn largest_error_with_momenta(errors: [f64; 8], momenta: usize) -> f64 {
        const MOMENTA: [usize; 8] = [0, 1, 1, 1, 2, 2, 2, 3];
        errors
            .into_iter()
            .zip(MOMENTA)
            .filter(|(_, count)| *count == momenta)
            .map(|(error, _)| error)
            .fold(0.0, f64::max)
    }

    #[test]
    fn sourced_transport_follows_the_subhorizon_initial_condition() {
        // Evolve half an e-fold inside the horizon and compare with the
        // analytic initial condition re-evaluated on the later slice. Blocks
        // with an odd number of momenta keep only their first surviving order,
        // so their truncation is relative O(aH/k) instead of O((aH/k)^2).
        let benchmark = sourced_benchmark([1.0, 0.9, 0.7]);
        let depth = 3.5_f64.exp();
        let later = benchmark.slice(depth);
        let evolved = benchmark.evolve(4.0, later, 1.0e-8);
        let formula = crate::canonical_three_point_initial_condition(
            &benchmark.system,
            &crate::BackgroundNeighborhood::from_trajectory(&benchmark.trajectory, later, 1.0e-3),
            benchmark.triangle,
            depth * (1.0 - 1.0e-6),
        )
        .unwrap();
        let errors = three_point_error(&evolved.alpha, &formula);

        assert!(largest_error_with_momenta(errors, 0) < 5.0e-3, "{errors:?}");
        assert!(largest_error_with_momenta(errors, 2) < 5.0e-3, "{errors:?}");
        assert!(largest_error_with_momenta(errors, 1) < 2.0e-2, "{errors:?}");
        assert!(largest_error_with_momenta(errors, 3) < 2.0e-1, "{errors:?}");

        // The tree-level velocity output leaves QQQ unchanged.
        let velocity = crate::velocity_three_point_from_transport_state(
            &benchmark.system,
            &benchmark.trajectory.state_at(later),
            benchmark.triangle,
            &evolved,
        )
        .unwrap();
        assert_eq!(velocity.qqq, evolved.alpha.qqq);
    }

    /// Flat field space in sheared coordinates `chi`, related to Cartesian
    /// `phi` by `phi^1 = chi^1` and `phi^2 = chi^2 + c (chi^1)^2`.
    ///
    /// The metric is position dependent and `Gamma^2_11 = 2c` is the only
    /// nonzero connection component, while the curvature vanishes. Running
    /// the same physical model in these coordinates exercises every
    /// connection term of the transport without changing the physics.
    #[derive(Clone, Copy, Debug)]
    struct ShearedFlatFieldSpace {
        shear: f64,
    }

    impl crate::FieldSpaceGeometry<2> for ShearedFlatFieldSpace {
        fn metric(&self, position: &[Field; 2]) -> crate::FieldMetric<2> {
            let slope = 2.0 * self.shear * position[0].into_value();
            crate::FieldMetric::new([
                [
                    crate::Quantity::new(1.0 + slope * slope),
                    crate::Quantity::new(slope),
                ],
                [crate::Quantity::new(slope), crate::Quantity::new(1.0)],
            ])
        }

        fn inverse_metric(&self, position: &[Field; 2]) -> crate::InverseFieldMetric<2> {
            let slope = 2.0 * self.shear * position[0].into_value();
            crate::InverseFieldMetric::new([
                [crate::Quantity::new(1.0), crate::Quantity::new(-slope)],
                [
                    crate::Quantity::new(-slope),
                    crate::Quantity::new(1.0 + slope * slope),
                ],
            ])
        }

        fn christoffel(&self, _position: &[Field; 2]) -> crate::ChristoffelSymbols<2> {
            let mut components = [[[crate::Quantity::<crate::MMinus1, f64>::new(0.0); 2]; 2]; 2];
            components[1][0][0] = crate::Quantity::new(2.0 * self.shear);
            crate::ChristoffelSymbols::new(components)
        }
    }

    impl crate::CurvedFieldSpaceGeometry<2> for ShearedFlatFieldSpace {
        fn riemann(&self, _position: &[Field; 2]) -> crate::RiemannTensor<2> {
            crate::RiemannTensor::zero()
        }
    }

    impl crate::ThirdOrderFieldSpaceGeometry<2> for ShearedFlatFieldSpace {
        fn christoffel_derivative(
            &self,
            _position: &[Field; 2],
        ) -> crate::ChristoffelDerivatives<2> {
            crate::ChristoffelDerivatives::zero()
        }
    }

    impl crate::CurvatureDerivativeGeometry<2> for ShearedFlatFieldSpace {
        fn covariant_riemann_derivative(
            &self,
            _position: &[Field; 2],
        ) -> crate::CovariantRiemannDerivative<2> {
            crate::CovariantRiemannDerivative::zero()
        }
    }

    /// The double-quadratic potential written in the sheared coordinates.
    #[derive(Clone, Copy, Debug)]
    struct ShearedQuadraticPotential {
        masses: [f64; 2],
        shear: f64,
    }

    impl<V> crate::MultifieldPotentialValue<V, 2> for ShearedQuadraticPotential
    where
        V: From<f64>
            + Clone
            + core::ops::Add<V, Output = V>
            + core::ops::Mul<V, Output = V>
            + core::ops::Mul<f64, Output = V>,
    {
        fn value(&self, chi: [Field<V>; 2]) -> crate::EnergyDensity<V> {
            let [first, second] = chi.map(crate::Quantity::into_value);
            let cartesian_second = second + first.clone() * first.clone() * self.shear;
            let [m1, m2] = self.masses;
            crate::EnergyDensity::new(
                first.clone() * first * (0.5 * m1 * m1)
                    + cartesian_second.clone() * cartesian_second * (0.5 * m2 * m2),
            )
        }
    }

    const SHEAR: f64 = 0.05;

    /// Cartesian and sheared descriptions of the same benchmark.
    fn coordinate_pair() -> (
        SourcedBenchmark,
        SourcedBenchmark<ShearedQuadraticPotential, ShearedFlatFieldSpace>,
    ) {
        let masses = [1.0e-6, 1.5e-6];
        let cartesian = sourced_benchmark([1.0, 0.9, 0.7]);
        let start = slow_roll_start(&cartesian.system, [12.0, 8.0]);
        let [phi1, phi2] = start.phi.map(Field::into_value);
        let [pi1, pi2] = start.pi.map(FieldVelocity::into_value);
        let sheared_system = CovariantMultifieldSystem::in_planck_units(
            ShearedQuadraticPotential {
                masses,
                shear: SHEAR,
            },
            ShearedFlatFieldSpace { shear: SHEAR },
        );
        let sheared_start = MultifieldState::new(
            [Field::new(phi1), Field::new(phi2 - SHEAR * phi1 * phi1)],
            [
                FieldVelocity::new(pi1),
                FieldVelocity::new(pi2 - 2.0 * SHEAR * phi1 * pi1),
            ],
            start.a,
        );
        let pivot = cartesian.triangle.canonical_wavenumbers()[0];
        let sheared = sourced_benchmark_with(
            sheared_system,
            sheared_start,
            [1.0, 0.9, 0.7],
            Some(pivot),
            &quick_settings(),
        );
        (cartesian, sheared)
    }

    /// Jacobian `d chi^I / d phi^J` at a Cartesian background state.
    fn shear_jacobian(state: &MultifieldState<2>) -> [[f64; 2]; 2] {
        [[1.0, 0.0], [-2.0 * SHEAR * state.phi[0].into_value(), 1.0]]
    }

    fn transform_three_point_block<A, B, C, D: Copy>(
        block: &crate::ThreePointBlock<A, B, C, D, Complex64, 2>,
        jacobian: &[[f64; 2]; 2],
    ) -> crate::ThreePointBlock<A, B, C, D, Complex64, 2> {
        let components = block.components();
        crate::ThreePointBlock::new(core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                core::array::from_fn(|k| {
                    let value: Complex64 = (0..8)
                        .map(|n| {
                            let [l, m, q] = [n / 4, (n / 2) % 2, n % 2];
                            jacobian[i][l]
                                * jacobian[j][m]
                                * jacobian[k][q]
                                * *components[l][m][q].value()
                        })
                        .sum();
                    crate::Quantity::new(value)
                })
            })
        }))
    }

    /// Transforms every canonical block as a rank-three contravariant tensor.
    fn transform_three_point(
        alpha: &crate::CanonicalThreePointPhaseSpaceCorrelation<2>,
        jacobian: &[[f64; 2]; 2],
    ) -> crate::CanonicalThreePointPhaseSpaceCorrelation<2> {
        crate::ThreePointPhaseSpaceCorrelation {
            qqq: transform_three_point_block(&alpha.qqq, jacobian),
            qqp: transform_three_point_block(&alpha.qqp, jacobian),
            qpq: transform_three_point_block(&alpha.qpq, jacobian),
            pqq: transform_three_point_block(&alpha.pqq, jacobian),
            qpp: transform_three_point_block(&alpha.qpp, jacobian),
            pqp: transform_three_point_block(&alpha.pqp, jacobian),
            ppq: transform_three_point_block(&alpha.ppq, jacobian),
            ppp: transform_three_point_block(&alpha.ppp, jacobian),
        }
    }

    #[test]
    fn initial_condition_transforms_as_a_tensor_under_a_field_redefinition() {
        let (cartesian, sheared) = coordinate_pair();
        let slice = cartesian.slice(4.0_f64.exp());
        let depth = 4.0_f64.exp() * (1.0 - 1.0e-3);
        let initial = |benchmark_trajectory: &BackgroundTrajectory| {
            crate::BackgroundNeighborhood::from_trajectory(benchmark_trajectory, slice, 1.0e-3)
        };
        let alpha_cartesian = crate::canonical_three_point_initial_condition(
            &cartesian.system,
            &initial(&cartesian.trajectory),
            cartesian.triangle,
            depth,
        )
        .unwrap();
        let alpha_sheared = crate::canonical_three_point_initial_condition(
            &sheared.system,
            &initial(&sheared.trajectory),
            sheared.triangle,
            depth,
        )
        .unwrap();
        let jacobian = shear_jacobian(&cartesian.trajectory.state_at(slice));
        let errors = three_point_error(
            &alpha_sheared,
            &transform_three_point(&alpha_cartesian, &jacobian),
        );

        assert!(errors.iter().all(|error| *error < 1.0e-8), "{errors:?}");
        // Comparing components without the Jacobian fails by order one, so the
        // agreement above is a statement about the transformation.
        let untransformed = three_point_error(&alpha_sheared, &alpha_cartesian);
        assert!(
            untransformed.iter().all(|error| *error > 0.1),
            "{untransformed:?}"
        );
    }

    #[test]
    fn sourced_transport_transforms_as_a_tensor_under_a_field_redefinition() {
        let (cartesian, sheared) = coordinate_pair();
        let target = cartesian.slice(3.5_f64.exp());
        let evolved_cartesian = cartesian.evolve(4.0, target, 1.0e-8);
        let evolved_sheared = sheared.evolve(4.0, target, 1.0e-8);
        let jacobian = shear_jacobian(&cartesian.trajectory.state_at(target));
        let errors = three_point_error(
            &evolved_sheared.alpha,
            &transform_three_point(&evolved_cartesian.alpha, &jacobian),
        );

        assert!(errors.iter().all(|error| *error < 1.0e-7), "{errors:?}");
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn sourced_transport_to_output_transforms_as_a_tensor() {
        let (cartesian, sheared) = coordinate_pair();
        let output = cartesian.output;
        let evolved_cartesian = cartesian.evolve(3.0, output, 1.0e-9);
        let evolved_sheared = sheared.evolve(3.0, output, 1.0e-9);
        let jacobian = shear_jacobian(&cartesian.trajectory.state_at(output));
        let errors = three_point_error(
            &evolved_sheared.alpha,
            &transform_three_point(&evolved_cartesian.alpha, &jacobian),
        );
        assert!(errors.iter().all(|error| *error < 1.0e-8), "{errors:?}");

        // zeta is a scalar: its correlators must not depend on the coordinates.
        let project = |system_state: &crate::CanonicalThreePointTransportState<2>,
                       background: MultifieldState<2>,
                       sheared_coordinates: bool| {
            if sheared_coordinates {
                crate::curvature::curvature_correlators(
                    &sheared.system,
                    &background,
                    sheared.triangle,
                    system_state,
                    0.1,
                )
            } else {
                crate::curvature::curvature_correlators(
                    &cartesian.system,
                    &background,
                    cartesian.triangle,
                    system_state,
                    0.1,
                )
            }
            .unwrap()
        };
        let zeta_cartesian = project(
            &evolved_cartesian,
            cartesian.trajectory.state_at(output),
            false,
        );
        let zeta_sheared = project(&evolved_sheared, sheared.trajectory.state_at(output), true);
        let relative = |a: f64, b: f64| (a - b).abs() / b.abs();
        assert!(
            relative(
                zeta_sheared.power[0].into_value(),
                zeta_cartesian.power[0].into_value()
            ) < 1.0e-8
        );
        assert!(
            relative(
                zeta_sheared.bispectrum.into_value(),
                zeta_cartesian.bispectrum.into_value()
            ) < 1.0e-7
        );
    }

    #[test]
    fn sourced_transport_keeps_equilateral_symmetry_and_reality() {
        let benchmark = sourced_benchmark([1.0, 1.0, 1.0]);
        let target = benchmark.slice(3.25_f64.exp());
        let alpha = benchmark.evolve(3.5, target, 1.0e-8).alpha;

        assert!(
            three_point_error(&alpha.swap_first_second(), &alpha)
                .iter()
                .all(|e| *e < 1.0e-10)
        );
        assert!(
            three_point_error(&alpha.swap_second_third(), &alpha)
                .iter()
                .all(|e| *e < 1.0e-10)
        );
        // The connected tree-level correlator is real; the complex covariances
        // must not leak an imaginary part into it.
        let imaginary = three_point_error(&alpha, &alpha.conjugate())
            .into_iter()
            .fold(0.0, f64::max);
        assert!(imaginary < 1.0e-8, "imaginary part {imaginary:e}");
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn sourced_transport_converges_with_start_depth() {
        let benchmark = sourced_benchmark([1.0, 0.9, 0.7]);
        let reference = benchmark.evolve(8.0, benchmark.output, 1.0e-10).alpha;
        let error = |depth: f64| {
            three_point_error(
                &benchmark.evolve(depth, benchmark.output, 1.0e-10).alpha,
                &reference,
            )
            .into_iter()
            .fold(0.0, f64::max)
        };
        // The error oscillates with the start phase, so compare envelopes over
        // one e-fold windows rather than neighbouring depths.
        let shallow = [4.0, 4.5, 5.0].map(error).into_iter().fold(0.0, f64::max);
        let deep = [6.0, 6.5, 7.0].map(error).into_iter().fold(0.0, f64::max);

        assert!(deep < 1.0e-2, "deep envelope {deep:e}");
        assert!(deep < 0.5 * shallow, "{shallow:e} then {deep:e}");
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn sourced_transport_converges_with_tighter_tolerance() {
        let benchmark = sourced_benchmark([1.0, 0.9, 0.7]);
        let loose = benchmark.evolve(4.0, benchmark.output, 1.0e-7).alpha;
        let tight = benchmark.evolve(4.0, benchmark.output, 1.0e-9).alpha;
        let error = three_point_error(&loose, &tight)
            .into_iter()
            .fold(0.0, f64::max);

        assert!(error < 1.0e-5, "tolerance change {error:e}");
    }

    /// Maldacena's single-field bispectrum as a reduced `f_NL`, with
    /// `epsilon` and `eta = d ln epsilon / dN` at horizon crossing.
    fn maldacena_f_nl(k: [f64; 3], epsilon: f64, eta: f64) -> f64 {
        let [k1, k2, k3] = k;
        let kt = k1 + k2 + k3;
        let cubes = k1.powi(3) + k2.powi(3) + k3.powi(3);
        let mixed =
            k1 * k2 * k2 + k1 * k3 * k3 + k2 * k1 * k1 + k2 * k3 * k3 + k3 * k1 * k1 + k3 * k2 * k2;
        let squares = k1 * k1 * k2 * k2 + k1 * k1 * k3 * k3 + k2 * k2 * k3 * k3;
        let shape = eta / 8.0 * cubes + epsilon / 8.0 * (-cubes + mixed + 8.0 * squares / kt);
        // B = (2 pi)^4 Pcal^2 shape / prod k^3 and P = 2 pi^2 Pcal / k^3.
        let pairs =
            (k1 * k2).powi(3).recip() + (k1 * k3).powi(3).recip() + (k2 * k3).powi(3).recip();
        5.0 / 6.0 * 4.0 * shape / (k1 * k2 * k3).powi(3) / pairs
    }

    /// Settings with the pivot well inside slow roll and a superhorizon output.
    fn slow_roll_settings() -> SpectrumRunSettings {
        SpectrumRunSettings {
            e_folds_before_exit: 5.0,
            pivot_e_folds_before_end: 40.0,
            output_e_folds_before_end: 34.0,
            ..quick_settings()
        }
    }

    /// `epsilon` and `eta = d ln epsilon / dN` when `k_t / 3` crosses the
    /// horizon on the straight equal-mass trajectory.
    fn slow_roll_parameters_at_crossing<P, G>(bench: &SourcedBenchmark<P, G>) -> (f64, f64)
    where
        P: crate::ThriceDifferentiableMultifieldPotential<2> + Copy,
        G: crate::ThirdOrderFieldSpaceGeometry<2> + crate::CurvatureDerivativeGeometry<2> + Copy,
    {
        let average = Mass::new(bench.triangle.canonical_wavenumbers().iter().sum::<f64>() / 3.0);
        let crossing = bench
            .trajectory
            .crossing(|s| (average / (s.a * bench.system.hubble(s))).into_value() - 1.0)
            .unwrap();
        let epsilon = |n: f64| {
            let s = bench.trajectory.state_at(n);
            let h = bench.system.hubble(&s).into_value();
            (s.pi[0].into_value().powi(2) + s.pi[1].into_value().powi(2)) / (2.0 * h * h)
        };
        let step = 1.0e-3;
        let eta = (epsilon(crossing + step).ln() - epsilon(crossing - step).ln()) / (2.0 * step);
        (epsilon(crossing), eta)
    }

    fn project_benchmark<P, G>(
        bench: &SourcedBenchmark<P, G>,
        depth: f64,
        slice: f64,
    ) -> crate::curvature::CurvatureCorrelators
    where
        P: crate::ThriceDifferentiableMultifieldPotential<2> + Copy,
        G: crate::ThirdOrderFieldSpaceGeometry<2> + crate::CurvatureDerivativeGeometry<2> + Copy,
    {
        let state = bench.evolve(depth, slice, 1.0e-9);
        crate::curvature::curvature_correlators(
            &bench.system,
            &bench.trajectory.state_at(slice),
            bench.triangle,
            &state,
            0.1,
        )
        .unwrap()
    }

    #[test]
    fn curvature_power_from_transport_matches_the_mode_solver() {
        // The turning benchmark, with both solvers started at the same depth.
        let settings = SpectrumRunSettings {
            e_folds_before_exit: 3.0,
            ..quick_settings()
        };
        let turning = system([1.0e-6, 1.5e-6]);
        let initial = slow_roll_start(&turning, [12.0, 8.0]);
        let bench = sourced_benchmark_with(turning, initial, [1.0, 1.0, 1.0], None, &settings);
        let projected = project_benchmark(&bench, 3.0, bench.output);
        let k = Mass::new(bench.triangle.canonical_wavenumbers()[0]);
        let (row, _, _) = bench
            .system
            .evolve_one_mode(initial, &settings, &bench.trajectory, k, 1.0, bench.output)
            .unwrap();
        let transported = projected.dimensionless_power()[0];

        // zeta = -R, so the spectra agree without a sign.
        assert!(
            (transported - row.p_r).abs() <= 1.0e-5 * row.p_r,
            "{transported:e} != {:e}",
            row.p_r
        );
        assert!(projected.bispectrum_imaginary_ratio < 1.0e-6);
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn single_field_f_nl_is_conserved_and_matches_maldacena() {
        let settings = slow_roll_settings();
        let straight = system([1.0e-6, 1.0e-6]);
        let start = slow_roll_start(&straight, [12.0, 8.0]);
        for shape in [[1.0, 1.0, 1.0], [1.0, 1.0, 0.5], [1.0, 1.0, 0.2]] {
            let bench = sourced_benchmark_with(straight, start, shape, None, &settings);
            let late = project_benchmark(&bench, 5.0, bench.output).reduced_f_nl();
            let early = project_benchmark(&bench, 5.0, bench.output - 1.0).reduced_f_nl();
            let (epsilon, eta) = slow_roll_parameters_at_crossing(&bench);
            let leading = maldacena_f_nl(bench.triangle.canonical_wavenumbers(), epsilon, eta);

            assert!(
                (late - early).abs() < 1.0e-2 * late,
                "{shape:?}: {early} then {late}"
            );
            // Maldacena's result is leading order in slow roll; epsilon ~ 0.0125.
            assert!(
                (late - leading).abs() < 5.0e-2 * leading,
                "{shape:?}: {late} vs {leading}"
            );
        }
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn squeezed_single_field_bispectrum_satisfies_the_consistency_relation() {
        let settings = slow_roll_settings();
        let straight = system([1.0e-6, 1.0e-6]);
        let start = slow_roll_start(&straight, [12.0, 8.0]);
        let deviation = |ratio: f64| {
            let bench = sourced_benchmark_with(straight, start, [1.0, 1.0, ratio], None, &settings);
            let f_nl = project_benchmark(&bench, 5.0, bench.output).reduced_f_nl();
            let short = bench.triangle.canonical_wavenumbers()[0];
            let spectrum = |k: f64| {
                bench
                    .system
                    .evolve_one_mode(
                        start,
                        &settings,
                        &bench.trajectory,
                        Mass::new(k),
                        1.0,
                        bench.output,
                    )
                    .unwrap()
                    .0
                    .p_r
            };
            let step = 0.01_f64;
            let tilt = (spectrum(short * step.exp()).ln() - spectrum(short * (-step).exp()).ln())
                / (2.0 * step);
            let consistency = -5.0 / 12.0 * tilt;
            (f_nl - consistency) / consistency
        };
        let coarse = deviation(0.1);
        let fine = deviation(0.05);

        // Corrections to the consistency relation scale as (k_long / k_short)^2.
        assert!(fine.abs() < 1.0e-2, "relative deviation {fine:e}");
        assert!(fine.abs() < 0.5 * coarse.abs(), "{coarse:e} then {fine:e}");
    }

    /// The axion-quartic model of Elliston et al. (arXiv:1203.6844), with the
    /// parameters of Dias, Frazer, Mulryne, and Seery (arXiv:1609.00379, §9.1).
    #[derive(Clone, Copy, Debug)]
    struct AxionQuarticPotential {
        quartic: f64,
        axion_scale: f64,
        decay_constant: f64,
    }

    impl AxionQuarticPotential {
        fn dfms() -> Self {
            let quartic = 1.0e-10;
            Self {
                quartic,
                axion_scale: (25.0 / (2.0 * PI)).powi(2) * quartic,
                decay_constant: 1.0,
            }
        }
    }

    impl<V> crate::MultifieldPotentialValue<V, 2> for AxionQuarticPotential
    where
        V: From<f64>
            + Clone
            + core::ops::Add<V, Output = V>
            + core::ops::Sub<V, Output = V>
            + core::ops::Mul<V, Output = V>
            + core::ops::Mul<f64, Output = V>
            + crate::Transcendental,
    {
        fn value(&self, fields: [Field<V>; 2]) -> crate::EnergyDensity<V> {
            let [phi, chi] = fields.map(crate::Quantity::into_value);
            let squared = phi.clone() * phi;
            let angle = (chi * (2.0 * PI / self.decay_constant)).cos();
            crate::EnergyDensity::new(
                squared.clone() * squared * (self.quartic / 4.0)
                    + (V::from(1.0) - angle) * self.axion_scale,
            )
        }
    }

    type AxionSystem = CovariantMultifieldSystem<AxionQuarticPotential, FlatFieldSpace, 2>;

    fn axion_start(system: &AxionSystem, phi: f64, chi: f64) -> MultifieldState<2> {
        let resting = MultifieldState::new(
            [Field::new(phi), Field::new(chi)],
            [FieldVelocity::new(0.0); 2],
            ScaleFactor::new(1.0),
        );
        MultifieldState {
            pi: system.slow_roll_velocity(&resting),
            ..resting
        }
    }

    /// Samples of the equilateral reduced bispectrum in Fig. 7 of Dias, Frazer,
    /// Mulryne, and Seery (arXiv:1609.00379v2), read from the vector path of
    /// the published figure by `docs/derivations/digitize_dfms_figure7.py`.
    const DFMS_FIGURE7_F_NL: [(f64, f64); 9] = [
        (59.8400, -1.5825),
        (61.2000, -22.1643),
        (62.0160, -39.2245),
        (63.1040, -13.3209),
        (64.6000, 6.1135),
        (65.5520, 16.9603),
        (66.2320, 12.7325),
        (67.0480, 13.1427),
        (68.0000, 13.1162),
    ];

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn axion_quartic_bispectrum_matches_dfms_and_numerical_delta_n() {
        let system = CovariantMultifieldSystem::in_planck_units(
            AxionQuarticPotential::dfms(),
            FlatFieldSpace,
        );
        let initial = axion_start(&system, 23.5, 0.5 - 1.0e-3);
        let settings = SpectrumRunSettings {
            output_e_folds_before_end: 0.5,
            ..quick_settings()
        };
        let (_, trajectory, _) = system.survey_background(initial, &settings).unwrap();
        // As in DFMS, k_t/3 exits 14 e-folds after the initial time.
        let exit = trajectory.state_at(14.0);
        let k = (exit.a * system.hubble(&exit)).into_value();
        let bench = sourced_benchmark_with(system, initial, [1.0, 1.0, 1.0], Some(k), &settings);

        for (e_folds, published) in DFMS_FIGURE7_F_NL {
            let f_nl = project_benchmark(&bench, 5.0, e_folds).reduced_f_nl();
            assert!(
                (f_nl - published).abs() < 5.0e-3 * published.abs().max(10.0),
                "N={e_folds}: {f_nl} against the published {published}"
            );
        }

        // Numerical delta N from a flat slice six e-folds after exit to the
        // uniform-density surface of the output slice. It uses only the
        // background equations, so it is independent of the transport.
        let base = trajectory.state_at(20.0);
        let target_hubble = system.hubble(&trajectory.state_at(bench.output));
        let precise = SpectrumRunSettings {
            tolerances: ErrorTolerances::new(
                crate::ConformalPerturbationTolerance::<2>::background_in_planck_units(1.0e-24),
                1.0e-13,
            ),
            ..settings
        };
        let remaining = |shift: [f64; 2]| {
            let start = axion_start(
                &system,
                base.phi[0].into_value() + shift[0],
                base.phi[1].into_value() + shift[1],
            );
            let (_, perturbed, _) = system.survey_background(start, &precise).unwrap();
            perturbed
                .crossing(|state| (system.hubble(state) - target_hubble).into_value())
                .unwrap()
        };
        // The step is a small fraction of the axion's distance from its hilltop.
        let h = 1.0e-2 * (0.5 - base.phi[1].into_value());
        let value = |i: i32, j: i32| remaining([f64::from(i) * h, f64::from(j) * h]);
        let centre = value(0, 0);
        let first = [
            (value(1, 0) - value(-1, 0)) / (2.0 * h),
            (value(0, 1) - value(0, -1)) / (2.0 * h),
        ];
        let mixed = (value(1, 1) - value(1, -1) - value(-1, 1) + value(-1, -1)) / (4.0 * h * h);
        let second = [
            [(value(1, 0) - 2.0 * centre + value(-1, 0)) / (h * h), mixed],
            [mixed, (value(0, 1) - 2.0 * centre + value(0, -1)) / (h * h)],
        ];
        let contraction: f64 = (0..4)
            .map(|m| first[m / 2] * first[m % 2] * second[m / 2][m % 2])
            .sum();
        let norm = first[0] * first[0] + first[1] * first[1];
        let delta_n = 5.0 / 6.0 * contraction / (norm * norm);
        let transported = project_benchmark(&bench, 5.0, bench.output).reduced_f_nl();

        assert!(
            (transported - delta_n).abs() < 5.0e-3 * delta_n.abs(),
            "transport {transported} against delta N {delta_n}"
        );
    }
    /// `P_R`, `P_t`, and equilateral `f_NL` of one triangle at the pivot,
    /// for any number of fields.
    fn n_field_observables<P, G, const N: usize>(
        system: &CovariantMultifieldSystem<P, G, N>,
        initial: MultifieldState<N>,
        settings: &SpectrumRunSettings,
    ) -> (f64, f64, f64)
    where
        P: crate::ThriceDifferentiableMultifieldPotential<N> + Copy,
        G: crate::ThirdOrderFieldSpaceGeometry<N> + crate::CurvatureDerivativeGeometry<N> + Copy,
    {
        let survey = system.survey(initial, settings).unwrap();
        let output = survey.e_folds_at_end - settings.output_e_folds_before_end;
        let pivot = survey
            .trajectory
            .state_at(survey.e_folds_at_end - settings.pivot_e_folds_before_end);
        let k = pivot.a * system.hubble(&pivot);
        let scalar = system
            .mode_spectrum(settings, &survey.trajectory, k, output)
            .unwrap()
            .spectra
            .curvature
            .into_value();
        let tensor = system
            .tensor_spectrum(settings, &survey.trajectory, k, output)
            .unwrap();
        let bispectrum = system
            .run_bispectrum(
                initial,
                &crate::BispectrumRunSettings {
                    background: *settings,
                    ..crate::BispectrumRunSettings::default()
                },
                &[crate::TriangleShape::equilateral(3.0)],
            )
            .unwrap();
        (scalar, tensor, bispectrum.records[0].f_nl)
    }

    fn resting_slow_roll<P, G, const N: usize>(
        system: &CovariantMultifieldSystem<P, G, N>,
        phi: [f64; N],
    ) -> MultifieldState<N>
    where
        P: crate::MultifieldPotential<N>,
        G: crate::FieldSpaceGeometry<N>,
    {
        let resting = MultifieldState::new(
            phi.map(Field::new),
            [FieldVelocity::new(0.0); N],
            ScaleFactor::new(1.0),
        );
        MultifieldState {
            pi: system.slow_roll_velocity(&resting),
            ..resting
        }
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn one_field_runs_match_two_fields_with_a_decoupled_spectator() {
        let settings = SpectrumRunSettings {
            pivot_e_folds_before_end: 25.0,
            e_folds_before_exit: 4.0,
            ..quick_settings()
        };
        let single = CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(1.0e-6)]),
            FlatFieldSpace,
        );
        let pair = system([1.0e-6, 3.0e-5]);
        let one = n_field_observables(&single, resting_slow_roll(&single, [12.0]), &settings);
        let two = n_field_observables(&pair, resting_slow_roll(&pair, [12.0, 0.0]), &settings);
        assert!(
            (one.0 / two.0 - 1.0).abs() < 1.0e-9,
            "P_R {} vs {}",
            one.0,
            two.0
        );
        assert!(
            (one.1 / two.1 - 1.0).abs() < 1.0e-9,
            "P_t {} vs {}",
            one.1,
            two.1
        );
        assert!(
            (one.2 / two.2 - 1.0).abs() < 1.0e-6,
            "f_NL {} vs {}",
            one.2,
            two.2
        );
    }

    #[test]
    #[ignore = "expensive; run with `cargo test --release -- --ignored`"]
    fn three_field_runs_match_two_fields_with_a_decoupled_spectator() {
        let settings = SpectrumRunSettings {
            pivot_e_folds_before_end: 25.0,
            e_folds_before_exit: 4.0,
            ..quick_settings()
        };
        let pair = system([1.0e-6, 7.0e-6]);
        let triple = CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([
                Mass::new(1.0e-6),
                Mass::new(7.0e-6),
                Mass::new(3.0e-4),
            ]),
            FlatFieldSpace,
        );
        let two = n_field_observables(&pair, resting_slow_roll(&pair, [14.0, 12.0]), &settings);
        let three = n_field_observables(
            &triple,
            resting_slow_roll(&triple, [14.0, 12.0, 0.0]),
            &settings,
        );
        // The spectator changes the adaptive steps, so the agreement is at the
        // integration tolerance rather than bit for bit.
        assert!(
            (two.0 / three.0 - 1.0).abs() < 1.0e-8,
            "P_R {} vs {}",
            two.0,
            three.0
        );
        assert!(
            (two.1 / three.1 - 1.0).abs() < 1.0e-8,
            "P_t {} vs {}",
            two.1,
            three.1
        );
        assert!(
            (two.2 / three.2 - 1.0).abs() < 1.0e-6,
            "f_NL {} vs {}",
            two.2,
            three.2
        );
    }
}
