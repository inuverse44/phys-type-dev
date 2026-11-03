//! Dimension-aware physical quantities for numerical physics.
//!
//! The crate encodes physical dimensions in the type system so that
//! dimensionally invalid expressions fail to compile. See `docs/mvp-design.md`
//! for the design that this implements.

pub mod autodiff;
pub mod background;
pub mod bispectrum_run;
pub mod curvature;
pub mod dimension;
pub mod field_space;
pub mod integrate;
pub mod jet_geometry;
pub mod multifield;
pub mod observation;
pub mod perturbation;
pub mod potential;
pub mod quantity;
pub mod scalar_field;
pub mod spectra;
pub mod spectrum_run;
pub mod split;
pub mod state;
#[cfg(test)]
mod test_models;
pub mod theory;
pub mod time;
pub mod transport;

pub use autodiff::{Dual, FourthOrderJet, Jet, ThirdOrderJet, Transcendental};
pub use background::{
    AdaptiveRunSettings, AdaptiveStep, BackgroundRun, Record, RunSettings, Termination,
};
pub use bispectrum_run::{
    BispectrumRecord, BispectrumRun, BispectrumRunError, BispectrumRunSettings,
    PhaseSpaceBispectrumRecord, ThreePointEvolutionError, TriangleShape, write_bispectrum_metadata,
};
pub use curvature::{
    CurvatureCorrelators, CurvatureGaugeCoefficients, CurvatureGaugeQuadratic,
    CurvatureProjectionError, curvature_correlators, curvature_gauge_coefficients,
};
pub use dimension::{
    DimDiv, DimMul, DimSqrt, M0, M1, M2, M3, M3Over2, M4, M5Over2, MHalf, MMinus1, MMinus2,
    MMinus3, MMinus3Over2, MMinus6, MMinusHalf, MassPower,
};
pub use field_space::{
    ChristoffelDerivativeComponents, ChristoffelDerivatives, ChristoffelSymbols,
    ContravariantField, CovariantField, CovariantPotentialHessian,
    CovariantPotentialThirdDerivative, CovariantRiemannDerivative, CurvatureDerivativeGeometry,
    CurvedFieldSpaceGeometry, FieldMetric, FieldSpaceGeometry, FieldTensor2, FieldTensor3,
    FieldTensor4, FieldTensor5, FieldVector, FlatFieldSpace, GeometryDomainError,
    GeometryValidationError, HyperbolicFieldSpace, InvalidCurvatureRadius, InverseFieldMetric,
    KinematicBasis2, KinematicBasisError, Lower, LowerFirst, LoweredRiemannTensor, PolarFieldSpace,
    RaiseFirst, RiemannIdentity, RiemannTensor, Tensor3Components, Tensor4Components,
    Tensor5Components, ThirdOrderFieldSpaceGeometry, Upper, validate_riemann_symmetries,
};
pub use integrate::{
    AdaptiveAttempt, AdaptiveSystem, EmbeddedStep, ErrorTolerances, StepSizeLimits, System,
    adaptive_step, component_error_ratio, next_step_size, rk4_step, rk45_step,
};
pub use jet_geometry::{JetGeometry, MetricJet};
pub use multifield::{
    CovariantMultifieldSystem, Kinematics, MultifieldState, MultifieldStateDelta,
    MultifieldStateDerivative, MultifieldSystem, TwoFieldKinematics,
};
pub use num_complex::Complex64;
pub use observation::{
    AmplitudeRole, ComovingWavenumberMpcInverse, InvalidObservationalWavenumber,
    LikelihoodComponent, ObservationDataRelease, ObservationalReference,
    PLANCK_2018_TT_TE_EE_LOW_E, write_observational_metadata,
};
pub use perturbation::{
    ComplexMagnitude, ComplexMatrix, ConformalBackgroundState, ConformalBackgroundTolerance,
    ConformalPerturbationDelta, ConformalPerturbationDerivative, ConformalPerturbationState,
    ConformalPerturbationSystem, ConformalPerturbationTolerance, ModeAcceleration, ModeAmplitude,
    ModeInitializationError, ModeVelocity, Modes, PerturbationDelta, PerturbationDerivative,
    PerturbationState, PerturbationSystem, PerturbationTolerance, normalized_mode_error,
};
pub use potential::{
    MultifieldPotential, MultifieldPotentialValue, Potential, PotentialDerivatives,
    PotentialDerivativesThroughThird, PotentialValue, QuadraticMultifieldPotential,
    QuadraticPotential, ThriceDifferentiableMultifieldPotential,
    TwiceDifferentiableMultifieldPotential,
};
pub use quantity::{Quantity, Sqrt};
pub use scalar_field::ScalarFieldSystem;
pub use spectra::{
    DimensionlessSpectrum, FieldPowerSpectrum, InvariantSpectra, ModeCovariance, ModeProjections,
    ProjectedSpectra, SpectrumError, SpectrumTolerances, field_power_spectrum,
    hermiticity_residual, invariant_spectra, mode_covariance, project,
};
pub use spectrum_run::{
    BackgroundRecord, BackgroundSample, BackgroundSurvey, BackgroundTrajectory, EndOfInflation,
    ModeEvolutionRecord, ModeSpectrum, Pivot, SpectrumRecord, SpectrumRun, SpectrumRunError,
    SpectrumRunSettings, SpectrumRunStage, write_metadata,
};
pub use split::{CurvatureResponse, SplitError, SplitSettings};
pub use state::{State, StateDelta, StateDerivative};
pub use theory::{
    EinsteinFrame, EinsteinGeometry, EinsteinMetric, EinsteinPotential, Formulation,
    FrameCorrespondence, GeneralizedNonminimal, TheoryScalar, einstein_frame,
};
pub use time::ConformalTime;
pub use transport::{
    AnyPhaseSpaceIndex, BackgroundNeighborhood, BackgroundProvider, CanonicalCubicACoefficients,
    CanonicalCubicBCoefficients, CanonicalCubicCCoefficients,
    CanonicalCubicHamiltonianCoefficients, CanonicalCubicHamiltonianVertices, CanonicalKernelError,
    CanonicalMomentumPerturbation, CanonicalQuadraticPhaseSpaceKernel,
    CanonicalSubhorizonInitialVertices, CanonicalThreePointPhaseSpaceCorrelation,
    CanonicalThreePointTransportState, CanonicalThreePointTransportSystem,
    CanonicalThreePointTransportTolerance, CanonicalTwoPointPhaseSpaceCovariance,
    CubicCoefficientError, DEFAULT_TRIANGLE_CLOSURE_TOLERANCE, FieldPerturbation, FourierLeg,
    MomentumPerturbation, MomentumTriangle, MomentumTriangleError, PhaseSpaceIndex,
    PhaseSpaceIndexError, PhaseSpaceSlot, PhaseSpaceSlotKind, QuadraticPhaseSpaceKernel,
    ThreePointBlock, ThreePointHomogeneousTransportSystem, ThreePointInitialConditionError,
    ThreePointPhaseSpaceCorrelation, ThreePointTransportTolerance, TwoPointBlock,
    TwoPointPhaseSpaceCovariance, TwoPointTransportSystem, TwoPointTransportTolerance,
    canonical_cubic_a_coefficients, canonical_cubic_b_coefficients, canonical_cubic_c_coefficients,
    canonical_cubic_hamiltonian_coefficients, canonical_cubic_hamiltonian_vertices,
    canonical_subhorizon_initial_vertices, canonical_subhorizon_vertex_rates,
    canonical_three_point_from_subhorizon_vertices, canonical_three_point_initial_condition,
    canonical_three_point_transport_initial_state, canonical_two_point_from_velocity_linear,
    three_point_source_from_kernels, two_point_covariance_from_mode_state,
    velocity_three_point_from_canonical_tree, velocity_three_point_from_transport_state,
    velocity_two_point_from_canonical_linear,
};

/// A dimensionless quantity, such as the scale factor.
pub type Dimensionless<V = f64> = Quantity<M0, V>;
/// The scale factor `a`.
pub type ScaleFactor<V = f64> = Quantity<M0, V>;
/// A scalar field value `phi`.
pub type Field<V = f64> = Quantity<M1, V>;
/// A scalar field velocity `dphi/dt`.
pub type FieldVelocity<V = f64> = Quantity<M2, V>;
/// The Hubble rate `H`.
pub type Hubble<V = f64> = Quantity<M1, V>;
/// An energy density, such as `V(phi)`.
pub type EnergyDensity<V = f64> = Quantity<M4, V>;
/// The gradient of a potential, `dV/dphi`.
pub type PotentialGradient<V = f64> = Quantity<M3, V>;
/// An entry in the coordinate or covariant Hessian of a potential.
pub type PotentialHessian<V = f64> = Quantity<M2, V>;
/// An entry in the third coordinate derivative of a potential.
pub type PotentialThirdDerivative<V = f64> = Quantity<M1, V>;
/// A mass, such as the reduced Planck mass.
pub type Mass<V = f64> = Quantity<M1, V>;
/// Cosmic time `t`.
pub type Time<V = f64> = Quantity<MMinus1, V>;
