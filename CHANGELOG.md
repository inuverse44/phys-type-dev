# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `phys_type::split`: `CovariantMultifieldSystem::split_curvature_correlators`
  stops the tree-level transport on a superhorizon slice and continues with
  the response of the final `zeta`, integrated backwards with the adjoint of
  the transport's own flows. Where the correlators grow during a turn and
  then cancel, the transport carried to the end keeps an error of the
  tolerance times that growth; the split does not, and its result does not
  depend on the slice. `phys-type-models/tests/kms_trajectories.rs` checks
  it on the three KMS trajectories (G4): trajectory 2 gives `f_NL = 38.287`,
  the same on two slices to `1e-6`, and the split reproduces the transport
  to `1e-5` where the transport converges.
- `phys-type-models/examples/kms_transport_probe.rs`, the diagnostics of
  issue #67. Its `gap` mode goes from the crossing-time delta N of KMS
  (`f_NL = 43.38` on trajectory 2) to the split (`38.287`) one ingredient
  at a time: the end surface (`epsilon_H = 1` against uniform `H`) moves it
  by `-10.3`, the slow-roll velocities at crossing by `-2.0`, and the
  massless statistics at horizon exit by `+7.2`.
- `CovariantMultifieldSystem::curvature_response` returns the response of
  the final `zeta` to the phase space on a slice (`CurvatureResponse`), the
  backward half of `split_curvature_correlators`.

- `scripts/chen_wang_squeezed.py` evaluates the squeezed coefficient `s(nu)`
  of quasi-single-field inflation (arXiv:0911.3380, Eq. (5.11)); the
  weak-mixing limit of the in-in reference reproduces it for
  `0.4 <= nu <= 1.2` (`tests/in_in_vs_paper.rs`).
- `scripts/hmps2_squeezed.py` evaluates the squeezed closed forms and the
  exact bispectrum representations of arXiv:2607.14529.
  `tests/in_in_vs_paper.rs` checks them against the in-in reference: the
  exact representations agree to `1e-8`, and the squeezed closed forms agree
  once their prefactor carries the factor 10/9 that the paper omits
  (`docs/literature-discrepancies.md` entry 12).
- `phys-type-models/tests/constant_turn_amplitudes.rs`: the transport
  bispectrum on the constant-turn orbit against the in-in reference of its
  decoupling-limit cubic action, equilateral and squeezed, to `5e-5` in
  `f_NL`. `examples/constant_turn_squeezed.rs` writes the in-in value as a
  new `f_nl_in_in` column (golden data refrozen; the transport columns are
  unchanged), and the figure shows it with the relative difference.
- `phys_type_models::in_in`, a numerical in-in reference for the
  constant-mixing two-field theory of arXiv:2607.14529, independent of the
  engine: modes from the paper's Eqs. (24)-(25) with a Bunch-Davies vacuum
  from the asymptotic series of the coupled equations, tree-level bispectra
  of `sigma phidot^2`, `sigma (nabla phi)^2`, `sigma^2 phidot`, `sigma^3`,
  their covariant forms, and `phidot^3`, a smooth window for `i0+`, and
  exact superhorizon commutators. `orbit_terms` gives the decoupling-limit
  cubic action of the constant-turn orbit. Validated against the exact power
  spectrum, the closed-form `phidot^3` bispectrum, and the paper's basis
  identities (`tests/in_in.rs`).
- `phys_type_models::hmps`, the closed-form power spectra of
  arXiv:2606.18248, moved from the constant-turn tests.
- `phys-type-models/tests/separable.rs`: three uncoupled quadratic fields
  against the sum-separable slow-roll `delta N` of Battefeld and Easther
  (arXiv:astro-ph/0610296, Eqs. (3) and (35)) and against phase-space
  `delta N` with the separate-universe lapse. This completes the `N = 3`
  validation of step 5.
- The assisted phase-space `delta N` starts each perturbed universe from the
  proper-time velocity and forms `delta phi = Q - Gamma Q Q / 2`; it now
  matches the transport to `1.3e-4` (was `2e-3`), and its tests assert
  `3e-4`.
- `phys_type_models::higgs::HiggsTheory`, single-field Higgs inflation in the
  metric and Palatini formulations, with first-order slow-roll predictions;
  `examples/higgs_inflation.rs`, `plots/higgs_inflation.py`, and the tracked
  figure `docs/figures/higgs_inflation.png`. The exact `n_s` and `r` follow
  slow roll to second-order accuracy and reach the large-`xi` limits.
- The Higgs example evaluates `N_* = 50` and 60 (new `pivot_e_folds` column).
  `examples/higgs_ns_r.rs` runs the same evolution on a grid of
  `log10 xi = -5..4` and `N_* = 50..60`; `plots/higgs_ns_r.py` overlays it,
  coloured by `log10 xi` in the layout of arXiv:2112.05283 Fig. 4, on the
  Planck 2018 and BICEP/Keck 2018 `n_s`-`r` contours (arXiv:2110.00483),
  fetched and checksum-verified by `scripts/fetch_bk18.py` into the untracked
  `data/`; tracked figure `docs/figures/higgs_ns_r.png`.
- `scripts/golden.py --only EXAMPLE` checks or freezes a single example.
- `CovariantMultifieldSystem::tensor_spectrum`: the tensor power spectrum of
  one wavenumber, evolved from the Bunch-Davies vacuum on a surveyed
  trajectory. It reproduces the second-order slow-roll `P_t` and the
  consistency relation `r = 16 epsilon_1 (1 + C epsilon_2)` in single-field
  quadratic inflation.
- The mode solver for any number of fields: `PerturbationSystem<P, G, N>`,
  `ConformalPerturbationSystem<P, G, N>`, `Modes<N>`, `PerturbationState<N>`,
  and their tolerances, with a Bunch-Davies frame built by Gram-Schmidt from
  the background tangent.
- `InvariantSpectra<N>` and `invariant_spectra`: the curvature spectrum, the
  total and principal entropy spectra, the norm of the curvature-entropy
  cross vector, and `r_C`, without an entropy basis.
- `Kinematics<N>` and `CovariantMultifieldSystem::kinematics` for any number
  of fields: the unit tangent, the entropy projector
  `perp^IJ = G^IJ - e_sigma^I e_sigma^J`, the turn vector `D_t e_sigma`, and
  its norm. Checked against the oriented two-field basis and against a
  finite-difference covariant derivative on flat and curved three-field
  spaces.
- Engine API for analyses outside the crate: `CovariantMultifieldSystem::survey`
  with `BackgroundSurvey` and a public `BackgroundTrajectory` (`state_at`,
  `crossing`), `mode_spectrum`, and `three_point_state` with
  `ThreePointEvolutionError`.
- The nonminimally assisted and constant-turn validations now run in
  `phys-type-models/tests` against `AssistedTheory` and the public API.
- The `phys-type-models` workspace crate: nonminimally assisted inflation as
  `AssistedTheory` (loop, power-law, and hybrid inflatons) on the theory
  layer, the HKKPT slow-roll prescription (`hkkpt`), the constant-turn orbit,
  and the `assisted_inflation`, `constant_turn_spectra`, and
  `constant_turn_squeezed` examples.

### Changed

- The run APIs take any number of fields: `survey` returns a
  `BackgroundSurvey<N>` with `BackgroundSample<N>` rows and a
  `BackgroundTrajectory<N>` (both default to two fields), `mode_spectrum`
  returns a `ModeSpectrum<N>` with basis-free `InvariantSpectra<N>` (its
  `initial` argument is gone), and `tensor_spectrum`, `three_point_state`,
  and `run_bispectrum` are generic over `N`. `run_two_field_spectra` keeps
  the oriented two-field records and CSV files.
- The two- and three-point transport and the curvature projection take any
  number of fields: `TwoPointTransportSystem`, the three-point transport
  systems, and `CanonicalThreePointTransportState` gain `const N`,
  `TwoFieldBackgroundProvider` becomes `BackgroundProvider<N>`, and
  `curvature_correlators` is generic over `N`.
- Two-field names lose "TwoField" in one break: `TwoFieldModes` becomes
  `Modes<2>`, `TwoFieldPerturbationState` becomes `PerturbationState<2>`,
  `ConformalTwoFieldPerturbationSystem<P, G>` becomes
  `ConformalPerturbationSystem<P, G, 2>`, `ComplexMatrix2` becomes
  `ComplexMatrix<2>`, and likewise for the derivative, delta, tolerance, and
  conformal types; `ModeCovariance` and `FieldPowerSpectrum` take `N`.
- The oriented two-field basis is now `oriented_kinematics`; `kinematics`
  returns the general `Kinematics<N>`.
- The core no longer exports the assisted model or its paper's formulas:
  `NonminimalAssistedFieldSpace`, the `Nonminimal*Potential` types, and the
  `assisted_slow_roll` and `nonminimal` modules become test fixtures of the
  core. Use `phys_type_models::assisted::AssistedTheory` with
  `phys_type::einstein_frame` instead.
- `JetGeometry` evaluates each quantity at the lowest derivative order it
  needs (`MetricJet::metric_values` and `metric_second`), which makes the
  automatically differentiated geometry about as fast as a hand-written one.
- The theory layer: `GeneralizedNonminimal<N>` states a theory by its
  Jordan-frame action (`Omega^2(phi) R`, `k_IJ(phi)`, `V_J(phi)`, metric or
  Palatini), `einstein_frame` maps it exactly to the Einstein-frame potential
  and a `JetGeometry`, and `EinsteinFrame::correspondence` returns the
  Jordan-frame scale factor, e-folds, Hubble rate, and field velocities.
  `docs/frames.md` records the map, the correspondence, and the frame
  dependence of the multi-field curvature perturbation.
- `FieldSpaceGeometry::check_domain`, checked by the background survey on
  every accepted step; `SpectrumRunError::OutsideGeometryDomain` reports a
  non-positive conformal factor or a metric that is not positive definite.
- `FourthOrderJet<N>`, forward-mode derivatives through fourth order, with
  products and compositions written as sums over index subsets and set
  partitions.
- `MetricJet` and `JetGeometry`: any metric supplied with three derivatives
  yields its inverse, connection, Riemann tensor, and covariant Riemann
  derivative by automatic differentiation. It reproduces the polar,
  hyperbolic, and nonminimally assisted geometries to `1e-12` and satisfies
  the second Bianchi identity on a generic three-field metric.
- A squeezed-bispectrum benchmark on the constant-turn orbit: the transport
  `f_NL(k, k, r k)` scales as `r^(3/2 - nu)` with `nu` set by the entropy mass
  `mu`, not by `m_eff`, as arXiv:2607.14529 states. Adds the ignored test
  `constant_turn_squeezed_bispectrum_scales_with_mu_not_the_effective_mass`,
  `examples/constant_turn_squeezed.rs`, `plots/constant_turn_squeezed.py`,
  and the tracked figure `docs/figures/constant_turn_squeezed.png`.
- `assisted_slow_roll`, the slow-roll delta N predictions of arXiv:2302.05866
  (Eqs. (3.1), (3.5), (3.9)) for the nonminimally assisted subclass, moved
  out of the tests so the example can share them.
- `examples/assisted_inflation.rs` and `plots/assisted_inflation.py`, with the
  tracked figure `docs/figures/assisted_inflation.png`: trajectories, `n_s`
  against `xi_2`, and `f_NL` against `n_s` for the loop, power-law, and
  hybrid examples, the counterpart of Figs. 9-10, 13-14, and 17-18 of the
  reference.
- `examples/constant_turn_spectra.rs` and `plots/constant_turn.py`, with the
  tracked figure `docs/figures/constant_turn_spectra.png`: mode-solver spectra
  on a constant-turn orbit over the exact solutions of arXiv:2606.18248.
- `PolarFieldSpace`, flat field space in polar coordinates, with
  finite-difference checks of its connection and curvature.
- A strong-mixing benchmark: on a constant-turn circular orbit the mode
  solver reproduces the exact power spectra of arXiv:2606.18248, Eqs.
  (43)-(45), to 3.5e-4 for mixing up to 2, and the massless Eq. (39) with
  `ln(1 / (2 z_end))` in place of its printed `ln(2 / z_end)`.
- `EndOfInflation` in `SpectrumRunSettings`: the background survey can end on
  a field-value crossing instead of `epsilon_H = 1`, with
  `SpectrumRunError::SlowRollEndedFirst` when slow roll ends first.
- `NonminimalPowerLawPotential` and `NonminimalHybridPotential`, the Class II
  examples of arXiv:2302.05866 Secs. 4.2 and 4.3. At the numerical `n_s =
  0.965` couplings, which all lie inside the quoted ranges, Eq. (3.1) matches
  the exact tilt, and the transport `f_NL` matches phase-space delta N.
- All four loop-inflation branches of arXiv:2302.05866 Sec. 4.1: given the
  exact end-of-inflation value of `s`, Eq. (3.1) matches the numerical tilt,
  and the transport `f_NL` matches a phase-space delta N.
- A phase-space delta N consistency test: on superhorizon slices, delta N
  with the evolved covariance and field three-point function reproduces the
  transport `f_NL` of assisted loop inflation to 0.05% for equilateral and
  squeezed shapes. The crossing-time delta N falls 20% short of that value.
- `NonminimalAssistedFieldSpace` and `NonminimalLoopInflationPotential`, the
  Einstein-frame model of nonminimally assisted inflation (arXiv:2302.05866)
  in `(phi, s)` coordinates. Their geometry is checked against
  finite-difference constructions. Its spectral index agrees with the
  reference's Eq. (3.1) once the exact end-of-inflation value of `s` is used.
- `CovariantMultifieldSystem::run_bispectrum`, which projects the tree-level
  bispectrum of Fergusson--Shellard `TriangleShape`s on a superhorizon output
  slice, with CSV writers for the projected rows and the canonical
  phase-space components and `#` metadata naming the model, gauge, sign,
  normalization, `f_NL` convention, shape, slices, and the absence of
  reheating. An isosceles grid test demonstrates shape resolution.
- A turning two-field benchmark: the axion-quartic model reproduces the
  equilateral `f_NL` of Fig. 7 of Dias, Frazer, Mulryne, and Seery at nine
  samples digitized from the figure's vector path, and a numerical delta N
  calculation agrees to 0.1%.
- The P3e curvature projection: covariant second-order gauge coefficients on
  a superhorizon slice, `P_zeta`, `B_zeta` with the operator-ordered gauge
  term, `k^3 P / (2 pi^2)`, and `f_NL` in the Maldacena/Planck convention.
  It is validated against the mode solver, superhorizon conservation,
  Maldacena's single-field result, the squeezed consistency relation, and a
  field redefinition. A `MMinus6` dimension alias supports `B_zeta`.
- A field-redefinition covariance test of the three-point initial condition
  and sourced evolution, repeating the benchmark in sheared flat coordinates
  with a position-dependent metric and constant connection.
- `CanonicalThreePointTransportSystem`, the adaptive sourced three-point
  evolution in the canonical basis, which co-evolves the three leg
  covariances and re-evaluates the model-derived vertices at every stage.
  Its initial state, velocity-basis output, block-wise dimensioned
  tolerances, and start-depth and tolerance convergence tests are included.
- First-order e-fold rates of the cubic tensors in the three-point initial
  condition, evaluated on a `BackgroundNeighborhood` with the covariant
  connection term. They remove an `O(epsilon)` start-depth floor of the odd
  momentum blocks.
- The P3d tree-level quantum initial condition for all eight canonical
  `QQQ`--`PPP` blocks on one common subhorizon slice, with separate fast-`A`,
  slow-`A`, leg-resolved `B`, and leg-resolved `C` contributions, a
  shortest-leg depth check, and a reproducible SymPy derivation in
  `docs/derivations/`.
- The first P3d three-point transport slice: a complete dimensioned
  `QQQ`--`PPP` phase-space correlation state, permutation-safe leg exchanges,
  complex conjugation, integration arithmetic, and compile-time rejection of
  block-dimension mismatches. Homogeneous three-leg evolution applies the P3c
  linear flow at each shared-background stage and preserves equal-leg
  permutation covariance. A dimensioned quadratic phase-space kernel and the
  three ordered complex `u Sigma Sigma` source contractions establish the
  hierarchy boundary before model-specific cubic coefficients are added.
- Separate horizon-exit spectrum rows refined at each mode's `k = aH`
  crossing. Both horizon-exit and requested-output CSVs carry `N`, `k/(aH)`,
  and explicit evaluation-epoch metadata.
- A typed Planck 2018 PR3 observational reference naming the baseline
  `TT,TE,EE+lowE` components, R3.00 likelihood release, and `0.05 Mpc^-1`
  scalar pivot independently of internal Planck-unit wavenumbers.
- CSV provenance that labels `A_s = 2.1e-9` as a benchmark target rather than
  a likelihood fit.
- A non-Gaussianity architecture selecting tree-level phase-space transport,
  with momentum-triangle, tensor, two-point-oracle, three-point, and nonlinear
  curvature-projection milestones.
- Validated, canonical momentum triangles with reversible caller-leg
  permutations, typed field/momentum phase-space indices, and dimensioned
  two-/three-point correlator blocks whose permutations preserve slot types.
- `ThirdOrderJet<N>` forward automatic differentiation, including third-order
  product and chain rules, and a dimensioned one-pass multifield potential API
  for the coordinate derivative `V_,IJK`.
- Third-order field-space geometry with explicit non-tensor `partial_M
  Gamma^I_JK`, fully symmetrized `V_;(IJK)`, and a typed rank-five
  `nabla_M R^I_JKL` capability. Flat and hyperbolic built-in geometries provide
  analytic implementations.
- A documented two-point transport convention `X = (Q, D_t Q)` and a typed
  mode-solver oracle producing complex `QQ`, `QP`, `PQ`, and `PP` covariance
  blocks with their distinct physical dimensions.
- Direct two-point covariance evolution on the shared cubic-Hermite background,
  including coordinate-basis connection flow on both tensor indices and
  all-block comparisons at subhorizon, horizon-exit, maximum-turn, and final
  epochs.
- Dimensioned adaptive tolerances for each two-point phase-space block, with
  tighter-tolerance and deeper-vacuum-start convergence tests completing the
  P3c two-point transport milestone.

### Fixed

- The three-point `u Sigma Sigma` source now keeps the operator order of
  `<X^a X^b X^c>`: covariances on legs to the left of the quadratic insertion
  contract as `Sigma^{ad}` rather than `Sigma^{da}`. The same correction
  applies to the tree-level canonical-to-velocity conversion.
- The canonical kernel map now exchanges momenta together with field indices
  in the `q_qp`, `p_pq`, and `p_pp` blocks and in the velocity redefinition.
  `CanonicalCubicHamiltonianCoefficients` and the `B`/`C` coefficient types
  gained separately evaluated input-exchanged tensors.

### Changed

- P3 documentation now uses rendered TeX for the transport hierarchy and
  geometry conventions, defines every index and momentum argument near first
  use, and separates implemented three-point infrastructure from the missing
  physical cubic kernel, quantum initial condition, and curvature projection.
- `canonical_three_point_initial_condition` and
  `canonical_three_point_from_subhorizon_vertices` now take the background
  neighborhood and the vertex rates, respectively.
- The linear two-point and homogeneous three-point flows are shared between
  the velocity and canonical-momentum markers.
- `k/k_pivot` is the canonical spectrum coordinate. Raw comoving wavenumbers
  are labeled `k_internal_over_m_pl` because they depend on the arbitrary
  scale-factor normalization.
- Spectrum runs now integrate the background once, store every accepted state,
  and evolve each mode independently against cubic-Hermite interpolation in
  elapsed e-folds. The former coupled driver remains a numerical test oracle.

## [0.2.0] - 2026-09-22

### Added

- Fixed-size field-space vectors and rank-two tensors with compile-time upper
  and lower indices.
- Explicit metric raise/lower operations and opposite-variance contraction.
- Compile-fail coverage for invalid field-space index operations.
- Covariant multifield background equations with pluggable metric, inverse
  metric, and Christoffel symbols.
- An oriented two-field adiabatic/entropy basis and signed turn rate.
- Second-order `Jet<N>` automatic differentiation for a potential's value,
  complete gradient, and Hessian in one evaluation.
- A shared `Transcendental` interface for `exp`, `ln`, `cos`, and `tanh` on
  `f64`, `Dual`, and `Jet<N>`, including second-order chain rules.
- Covariant potential Hessians including the Christoffel correction.
- Dense rank-three and rank-four field-space tensors with typed variance and
  physical dimension.
- Slot-specific first-index raising/lowering for higher-rank tensors, with
  compile-fail coverage for variance errors and connection/tensor confusion.
- A convention-fixed `RiemannTensor`, analytic curved-geometry capability,
  typed velocity contraction, and structured Levi-Civita symmetry validation.
- A complete covariant linear mass matrix including potential, Riemann, and
  gravitational contributions.
- Joint background and complex two-mode evolution with dimension-safe Fourier
  amplitudes, Bunch-Davies initialization, and Wronskian diagnostics.
- Dimensioned `Complex64` mode values, native conjugation/Hermitian algebra,
  and a modulus-based normalized error whose absolute tolerance keeps the
  physical dimension of the component it bounds.
- Hermitian mode covariance, `k^3 / (2 pi^2)` field power spectra, and
  adiabatic/entropy projections yielding `P_R`, `P_S`, signed `C_RS`, and a
  correlation coefficient, each guarded by explicit reality and range
  tolerances.
- An `AdaptiveSystem` error norm and a reusable Dormand-Prince step-size
  controller, implemented by the single-field background, both multifield
  backgrounds, and the two-field perturbation system.
- A two-field spectrum driver covering the background survey, the
  `k_pivot = a_* H_*` convention, per-wavenumber Bunch-Davies initialization at
  `x_init`, CSV rows with reproducibility metadata, and an example that writes
  the background, spectra, and mode-evolution files.
- A four-panel two-field figure, generated by the same one-command workflow as
  the single-field diagnostics, with run provenance read from the CSV metadata
  block and captioned on the figure.
- A focused final-inflation spectrum figure with `P_R(k)` and `P_S(k)` on a
  shared log-log axis and the pivot correlation annotated.
- `TwoFieldPerturbationSystem::mode_projections` reports the curvature and
  entropy amplitude of each independent mode, which the super-horizon transfer
  matrix needs.
- `HyperbolicFieldSpace`, a constant-negative-curvature two-field geometry, so
  that the covariant machinery is exercised with a non-vanishing connection and
  Riemann tensor rather than only through `FlatFieldSpace`.
- A semantically distinct `ConformalTime`, conformal background state storing
  `ln a`, and the drag-free `D_eta^2 v` two-field mode system, with
  compile-fail coverage preventing cosmic and conformal time from being mixed.

### Changed

- Two-field modes are integrated as `v = a U` rather than `U`, so the absolute
  error tolerance no longer has to sit below decaying mode amplitudes.
  `TwoFieldModes` exposes `rescaled_amplitude` and `rescaled_velocity`, and
  `mode_covariance` takes the complete state so the scale factor that undoes
  the rescaling cannot be mismatched with the modes.
- A spectrum run starts each mode a stated number of e-folds before its own
  horizon exit, replacing the raw depth `x_init`, which is now derived and
  still recorded in the CSV metadata.
- Absolute error tolerances are per group of components and carry each group's
  dimension. `AdaptiveSystem` gains an `AbsoluteTolerance` associated type and
  `ErrorTolerances` is generic over it. Mode components default to no absolute
  floor, because their error is normalized by a modulus that does not vanish.
- `Im_P_12` is emitted alongside `Re_P_12`, so both columns of a complex
  quantity cross the CSV boundary.
- Spectrum mode evolution now uses conformal time and stores `D_eta v`; local
  cosmic-time step settings are converted with `d eta = dt/a` at each
  attempt. The background survey remains in cosmic time.

### Fixed

- A spectrum run refuses a wavenumber that has not left the horizon by the
  requested output time, instead of reporting its Bunch-Davies vacuum as a
  spectrum, which for the pivot mode was twenty-one times the converged value.
- Event crossings are refined with the fifth-order Dormand-Prince state rather
  than a fourth-order sub-step. The state a mode starts from, and the state the
  spectra are read at, had never passed error control, which moved the reported
  spectra by `2e-5` to `5e-5`.
- `component_error_ratio` carries the component's dimension on every argument,
  so a background tolerance meant for one quantity cannot bound another. The
  complex mode path had always done this; the real path beside it had only the
  claim, made in three places.
- Mode step errors are normalized by the largest modulus in the mode matrix
  rather than by each component's own. An axis-aligned trajectory puts exact
  zeros off the diagonal, and such a component would take an unbounded ratio as
  soon as rounding made its error non-zero.
- The imaginary part of the off-diagonal spectrum entry is checked against the
  reality tolerance instead of being discarded unexamined.
- Spectrum-run settings are validated for finite values, physical ranges,
  ordered step sizes, and meaningful integration and projection tolerances
  before numerical work begins.
- Spectrum normalization takes the complete perturbation state, preventing a
  rescaled mode matrix from being paired with the scale factor of another time.
- Step-limit errors retain the run stage and wavenumber and implement the
  standard `Error` interface.
- `HyperbolicFieldSpace` rejects non-positive and non-finite curvature radii at
  construction and no longer exposes the validated field for mutation.

## [0.1.0] - 2026-09-21

### Added

- Compile-time mass dimensions with half-integer exponents.
- Dimension-aware quantities, states, derivatives, and state changes.
- Forward-mode automatic differentiation for potential gradients.
- Fixed-step RK4 and adaptive Dormand–Prince 5(4) integration.
- Single-field and canonical const-generic multifield backgrounds.
- Compile-fail tests for dimensionally invalid expressions.
- CSV diagnostics and reproducible Matplotlib visualization workflow.

[Unreleased]: https://github.com/inuverse44/phys-type/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/inuverse44/phys-type/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/inuverse44/phys-type/tree/v0.1.0
