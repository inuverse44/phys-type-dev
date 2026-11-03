//! Probe for issue #67: where does the tree-level transport f_NL of KMS
//! trajectory 2 lose precision, and what does phase-space delta N from an
//! earlier slice give? `<index>` is 0, 1, 2 for trajectories 1--3; slices
//! are given in e-folds before the end of inflation.
//!
//! - `transport <index> <tol,...> [<slice,...>]`: P_zeta and f_NL of the
//!   equilateral pivot triangle on a list of slices (decreasing), once per
//!   relative tolerance.
//! - `delta-n <index> <tol> <target> <n_c,...> <phi,chi,velocity>`:
//!   phase-space delta N from each slice `n_c` to the surface of uniform
//!   `H` at `target`, with the transport (tolerance `tol`) up to `n_c`; the
//!   last argument gives the finite-difference steps relative to each
//!   coordinate. The last row is the transport itself at `target`.
//! - `response <index> <tol> <target> <n_c,...> <step>`: the response of
//!   zeta on the surface `target` to the phase space on a slice, integrated
//!   backwards from `target` with RK4 steps of at most `step` e-folds,
//!   combined with the transport (tolerance `tol`) up to each slice `n_c`.
//!   This calls `CovariantMultifieldSystem::split_curvature_correlators`.
//! - `gap <index> <target> <delta,...>`: the gap between the transport and
//!   the crossing-time delta N of KMS Sec. V, one ingredient at a time. The
//!   slices are `delta` e-folds after the pivot crosses the horizon, and
//!   `target` is the output surface of uniform `H`.
//! - `noise <index> <n_c> <target>`: `N` along `chi` from `n_c`, to the `H`
//!   surface and to `epsilon_H = 1`, to measure the noise of the endpoint.
//!
//! `KMS_BACKGROUND_TOLERANCE` sets the relative tolerance of the background
//! survey (default `1e-10`); the response needs `1e-13` or below.
//!
//! ```sh
//! cargo run --release --example kms_transport_probe -- transport 1 1e-10,1e-13
//! cargo run --release --example kms_transport_probe -- delta-n 1 1e-12 0.01 50 1e-4,1e-3,1e-2
//! KMS_BACKGROUND_TOLERANCE=1e-13 \
//!     cargo run --release --example kms_transport_probe -- response 1 1e-12 0.01 10,50 1.25e-4
//! ```

use phys_type::{
    BackgroundNeighborhood, BackgroundSurvey, BackgroundTrajectory,
    CanonicalThreePointTransportState, CanonicalThreePointTransportSystem,
    CanonicalThreePointTransportTolerance, ConformalPerturbationTolerance, CurvatureResponse,
    Dimensionless, ErrorTolerances, Field, FieldSpaceGeometry, FieldVelocity, Formulation,
    MomentumTriangle, MultifieldState, ScaleFactor, SpectrumRunSettings, SplitSettings,
    StepSizeLimits, adaptive_step, canonical_three_point_transport_initial_state,
    curvature_correlators, next_step_size, velocity_three_point_from_transport_state,
};
use phys_type_models::kms::{
    KmsSystem, PHI_START, TRAJECTORIES, at_rest, benchmark_system, crossing_delta_n,
};

struct Transport<'a> {
    system: &'a KmsSystem,
    trajectory: &'a BackgroundTrajectory<2>,
    triangle: MomentumTriangle,
    state: CanonicalThreePointTransportState<2>,
    e_folds: f64,
    step: Dimensionless,
    tolerance: f64,
}

impl<'a> Transport<'a> {
    fn new(
        system: &'a KmsSystem,
        trajectory: &'a BackgroundTrajectory<2>,
        k: f64,
        tolerance: f64,
    ) -> Self {
        let triangle = MomentumTriangle::try_new(k, k, k).unwrap();
        let depth = 5.0_f64.exp();
        let start = trajectory
            .crossing(|s| k / (s.a * system.hubble(s)).into_value() - depth)
            .unwrap();
        let state = canonical_three_point_transport_initial_state(
            system,
            &BackgroundNeighborhood::from_trajectory(trajectory, start, 1.0e-3),
            triangle,
            depth * (1.0 - 1.0e-6),
        )
        .unwrap();
        Self {
            system,
            trajectory,
            triangle,
            state,
            e_folds: start,
            step: Dimensionless::new(1.0e-3),
            tolerance,
        }
    }

    fn advance(&mut self, target: f64) {
        let transport =
            CanonicalThreePointTransportSystem::new(self.system, self.trajectory, self.triangle);
        let tolerances = ErrorTolerances::new(
            CanonicalThreePointTransportTolerance::pure_relative(),
            self.tolerance,
        );
        let limits = StepSizeLimits {
            min: Dimensionless::new(1.0e-9),
            max: Dimensionless::new(5.0e-2),
        };
        while target - self.e_folds > 1.0e-13 {
            let attempted = Dimensionless::new(self.step.into_value().min(target - self.e_folds));
            let outcome = adaptive_step(
                &transport,
                Dimensionless::new(self.e_folds),
                &self.state,
                attempted,
                &tolerances,
            );
            if outcome.accepted {
                self.state = outcome.state;
                self.e_folds += attempted.into_value();
            }
            self.step = next_step_size(attempted, outcome.error_ratio, &limits);
        }
    }

    fn f_nl(&self) -> (f64, f64) {
        let background = self.trajectory.state_at(self.e_folds);
        let p = curvature_correlators(self.system, &background, self.triangle, &self.state, 0.1)
            .unwrap();
        (p.reduced_f_nl(), p.dimensionless_power()[0])
    }
}

fn covariance_entry(
    covariance: &phys_type::CanonicalTwoPointPhaseSpaceCovariance<2>,
    a: usize,
    b: usize,
) -> f64 {
    let (i, j) = (a % 2, b % 2);
    match (a < 2, b < 2) {
        (true, true) => covariance.qq.components()[i][j].value().re,
        (true, false) => covariance.qp.components()[i][j].value().re,
        (false, true) => covariance.pq.components()[i][j].value().re,
        (false, false) => covariance.pp.components()[i][j].value().re,
    }
}

fn three_point_entry(
    alpha: &phys_type::ThreePointPhaseSpaceCorrelation<2>,
    a: usize,
    b: usize,
    c: usize,
) -> f64 {
    let (i, j, k) = (a % 2, b % 2, c % 2);
    match (a < 2, b < 2, c < 2) {
        (true, true, true) => alpha.qqq.components()[i][j][k].value().re,
        (true, true, false) => alpha.qqp.components()[i][j][k].value().re,
        (true, false, true) => alpha.qpq.components()[i][j][k].value().re,
        (false, true, true) => alpha.pqq.components()[i][j][k].value().re,
        (true, false, false) => alpha.qpp.components()[i][j][k].value().re,
        (false, true, false) => alpha.pqp.components()[i][j][k].value().re,
        (false, false, true) => alpha.ppq.components()[i][j][k].value().re,
        (false, false, false) => alpha.ppp.components()[i][j][k].value().re,
    }
}

/// Phase-space delta N from the transport state at its current slice to the
/// surface `H = target_hubble`: (f_NL, f_NL Gaussian part, P_zeta).
fn phase_space_delta_n(
    transport: &Transport,
    settings: &SpectrumRunSettings,
    target_hubble: f64,
    scale: [f64; 3],
) -> (f64, f64, f64) {
    let system = transport.system;
    let base = transport.trajectory.state_at(transport.e_folds);
    let x0 = [
        base.phi[0].into_value(),
        base.phi[1].into_value(),
        base.pi[0].into_value(),
        base.pi[1].into_value(),
    ];
    let h = [
        scale[0] * x0[0].abs(),
        scale[1] * x0[1].abs(),
        scale[2] * x0[2].abs(),
        scale[2] * x0[3].abs(),
    ];
    let precise = SpectrumRunSettings {
        tolerances: ErrorTolerances::new(
            ConformalPerturbationTolerance::<2>::background_in_planck_units(1.0e-24),
            1.0e-13,
        ),
        background_sample_every: usize::MAX,
        ..*settings
    };
    let connection = system.geometry.christoffel(&base.phi);
    let hubble = system.hubble(&base).into_value();
    let at = |shift: [f64; 4]| {
        let mut point: [f64; 4] = core::array::from_fn(|a| x0[a] + shift[a]);
        for (i, field) in point.iter_mut().take(2).enumerate() {
            *field -= 0.5
                * (0..4)
                    .map(|m| {
                        let [j, kk] = [m / 2, m % 2];
                        connection.components()[i][j][kk].value() * shift[j] * shift[kk]
                    })
                    .sum::<f64>();
        }
        let state = |velocity: [f64; 2]| {
            MultifieldState::new(
                [Field::new(point[0]), Field::new(point[1])],
                velocity.map(FieldVelocity::new),
                ScaleFactor::new(1.0),
            )
        };
        let potential = 3.0 * system.hubble(&state([0.0; 2])).into_value().powi(2);
        let kinetic = 3.0
            * system
                .hubble(&state([point[2], point[3]]))
                .into_value()
                .powi(2)
            - potential;
        let lapse = ((3.0 * hubble * hubble - kinetic) / potential).sqrt();
        point[2] /= lapse;
        point[3] /= lapse;
        let start = MultifieldState::new(
            [Field::new(point[0]), Field::new(point[1])],
            [FieldVelocity::new(point[2]), FieldVelocity::new(point[3])],
            ScaleFactor::new(1.0),
        );
        let perturbed = system.survey(start, &precise).unwrap().trajectory;
        perturbed
            .crossing(|st| system.hubble(st).into_value() - target_hubble)
            .expect("a perturbed trajectory ended before the target surface")
    };
    let unit = |a: usize, sign: f64| -> [f64; 4] {
        core::array::from_fn(|b| if a == b { sign * h[a] } else { 0.0 })
    };
    let add = |u: [f64; 4], v: [f64; 4]| -> [f64; 4] { core::array::from_fn(|a| u[a] + v[a]) };
    let centre = at([0.0; 4]);
    let plus: [f64; 4] = core::array::from_fn(|a| at(unit(a, 1.0)));
    let minus: [f64; 4] = core::array::from_fn(|a| at(unit(a, -1.0)));
    let first: [f64; 4] = core::array::from_fn(|a| (plus[a] - minus[a]) / (2.0 * h[a]));
    let mut second = [[0.0; 4]; 4];
    for a in 0..4 {
        second[a][a] = (plus[a] - 2.0 * centre + minus[a]) / (h[a] * h[a]);
        for b in 0..a {
            let value = (at(add(unit(a, 1.0), unit(b, 1.0)))
                - at(add(unit(a, 1.0), unit(b, -1.0)))
                - at(add(unit(a, -1.0), unit(b, 1.0)))
                + at(add(unit(a, -1.0), unit(b, -1.0))))
                / (4.0 * h[a] * h[b]);
            second[a][b] = value;
            second[b][a] = value;
        }
    }

    let state = &transport.state;
    let velocity =
        velocity_three_point_from_transport_state(system, &base, transport.triangle, state)
            .unwrap();
    let mut map = [[0.0; 4]; 4];
    for (a, row) in map.iter_mut().enumerate() {
        row[a] = 1.0;
    }
    for i in 0..2 {
        for (kk, entry) in map[2 + i].iter_mut().take(2).enumerate() {
            *entry = -(0..2)
                .map(|j| connection.components()[i][j][kk].value() * x0[2 + j])
                .sum::<f64>();
        }
    }
    let sigma: [[[f64; 4]; 4]; 3] = core::array::from_fn(|leg| {
        core::array::from_fn(|a| {
            core::array::from_fn(|b| {
                (0..16)
                    .map(|m| {
                        let [c, d] = [m / 4, m % 4];
                        map[a][c] * map[b][d] * covariance_entry(&state.covariances[leg], c, d)
                    })
                    .sum()
            })
        })
    });
    let alpha = |a: usize, b: usize, c: usize| -> f64 {
        (0..64)
            .map(|m| {
                let [d, e, f] = [m / 16, (m / 4) % 4, m % 4];
                map[a][d] * map[b][e] * map[c][f] * three_point_entry(&velocity, d, e, f)
            })
            .sum()
    };
    let power: [f64; 3] = core::array::from_fn(|leg| {
        (0..16)
            .map(|m| first[m / 4] * first[m % 4] * sigma[leg][m / 4][m % 4])
            .sum()
    });
    let projected: [[f64; 4]; 3] = core::array::from_fn(|leg| {
        core::array::from_fn(|c| (0..4).map(|a| first[a] * sigma[leg][a][c]).sum())
    });
    let gaussian: f64 = [(1, 2), (0, 2), (0, 1)]
        .into_iter()
        .map(|(u, v)| {
            (0..16)
                .map(|m| projected[u][m / 4] * projected[v][m % 4] * second[m / 4][m % 4])
                .sum::<f64>()
        })
        .sum();
    let intrinsic: f64 = (0..64)
        .map(|m| {
            let [a, b, c] = [m / 16, (m / 4) % 4, m % 4];
            first[a] * first[b] * first[c] * alpha(a, b, c)
        })
        .sum();
    let pairs = power[0] * power[1] + power[0] * power[2] + power[1] * power[2];
    let f = |b: f64| 5.0 / 6.0 * b / pairs;
    let k = transport.triangle.canonical_wavenumbers()[0];
    (
        f(gaussian + intrinsic),
        f(gaussian),
        k.powi(3) * power[0] / (2.0 * core::f64::consts::PI.powi(2)),
    )
}

/// The delta N of KMS Sec. V from the fields `x0` with slow-roll velocities:
/// (f_NL, N_a N^a), with `N` read off each survey by `end`.
fn slow_roll_delta_n(
    system: &KmsSystem,
    x0: [f64; 2],
    settings: &SpectrumRunSettings,
    end: impl Fn(&BackgroundSurvey<2>) -> f64,
) -> (f64, f64) {
    // KMS_FD_STEP overrides the relative step `3e-4` of `crossing_delta_n`.
    let step = std::env::var("KMS_FD_STEP").map_or(3.0e-4, |value| value.parse().unwrap());
    let h = [step * x0[0].abs(), step * x0[1].abs()];
    let e_folds = |a: f64, b: f64| {
        let rest = at_rest(x0[0] + a * h[0], x0[1] + b * h[1]);
        let state = MultifieldState {
            pi: system.slow_roll_velocity(&rest),
            ..rest
        };
        end(&system.survey(state, settings).unwrap())
    };
    let centre = e_folds(0.0, 0.0);
    let axis = [
        [e_folds(1.0, 0.0), e_folds(-1.0, 0.0)],
        [e_folds(0.0, 1.0), e_folds(0.0, -1.0)],
    ];
    let mixed = (e_folds(1.0, 1.0) - e_folds(1.0, -1.0) - e_folds(-1.0, 1.0) + e_folds(-1.0, -1.0))
        / (4.0 * h[0] * h[1]);
    let first: [f64; 2] = core::array::from_fn(|a| (axis[a][0] - axis[a][1]) / (2.0 * h[a]));
    let mut second = [[mixed; 2]; 2];
    for a in 0..2 {
        second[a][a] = (axis[a][0] - 2.0 * centre + axis[a][1]) / (h[a] * h[a]);
    }
    let position = [Field::new(x0[0]), Field::new(x0[1])];
    let connection = system.geometry.christoffel(&position);
    let inverse = system.geometry.inverse_metric(&position);
    let raised: [f64; 2] = core::array::from_fn(|a| {
        (0..2)
            .map(|b| inverse.components()[a][b].into_value() * first[b])
            .sum()
    });
    let norm: f64 = (0..2).map(|a| raised[a] * first[a]).sum();
    let curvature: f64 = (0..4)
        .map(|m| {
            let [a, b] = [m / 2, m % 2];
            let covariant = second[a][b]
                - (0..2)
                    .map(|c| connection.components()[c][a][b].value() * first[c])
                    .sum::<f64>();
            raised[a] * raised[b] * covariant
        })
        .sum();
    (5.0 / 6.0 * curvature / (norm * norm), norm)
}

/// The response contracted with a covariance and a three-point function on
/// the phase space, equal legs: (f_NL, P up to a common factor, intrinsic
/// part of f_NL). `keep(a)` masks the phase-space directions.
fn contract(
    response: &CurvatureResponse,
    sigma: &dyn Fn(usize, usize) -> f64,
    alpha: &dyn Fn(usize, usize, usize) -> f64,
    keep: &dyn Fn(usize) -> bool,
) -> (f64, f64, f64) {
    let n: Vec<f64> = (0..4)
        .map(|a| if keep(a) { response.linear[0][a] } else { 0.0 })
        .collect();
    let power: f64 = (0..16)
        .map(|m| n[m / 4] * n[m % 4] * sigma(m / 4, m % 4))
        .sum();
    let projected: Vec<f64> = (0..4)
        .map(|a| (0..4).map(|c| n[c] * sigma(c, a)).sum())
        .collect();
    let gauge: f64 = (0..3)
        .map(|r| {
            (0..16)
                .map(|m| response.quadratic[r][m] * projected[m / 4] * projected[m % 4])
                .sum::<f64>()
        })
        .sum();
    let intrinsic: f64 = (0..64)
        .map(|m| n[m / 16] * n[(m / 4) % 4] * n[m % 4] * alpha(m / 16, (m / 4) % 4, m % 4))
        .sum();
    let f = |b: f64| 5.0 / 6.0 * b / (3.0 * power * power);
    (f(gauge + intrinsic), power, f(intrinsic))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args[1].as_str();
    let index: usize = args[2].parse().unwrap();
    let list = |s: &str| -> Vec<f64> { s.split(',').map(|x| x.parse().unwrap()).collect() };
    let system = benchmark_system(Formulation::Metric);
    // KMS_BACKGROUND_TOLERANCE overrides the relative tolerance of the survey.
    let background_tolerance =
        std::env::var("KMS_BACKGROUND_TOLERANCE").map_or(1.0e-10, |value| value.parse().unwrap());
    let settings = SpectrumRunSettings {
        pivot_e_folds_before_end: 60.0,
        e_folds_before_exit: 6.0,
        background_sample_every: 1,
        tolerances: ErrorTolerances::new(
            ConformalPerturbationTolerance::<2>::background_in_planck_units(
                1.0e-10 * background_tolerance,
            ),
            background_tolerance,
        ),
        ..SpectrumRunSettings::default()
    };
    let survey = system
        .survey(at_rest(PHI_START, TRAJECTORIES[index]), &settings)
        .unwrap();
    let end = survey.e_folds_at_end;
    let trajectory = &survey.trajectory;
    let pivot = trajectory.state_at(end - 60.0);
    let k = (pivot.a * system.hubble(&pivot)).into_value();
    match mode {
        "transport" => {
            let slices = args.get(4).map_or(
                vec![
                    57.0, 50.0, 40.0, 20.0, 10.0, 8.0, 7.0, 6.5, 6.0, 5.5, 5.2, 5.0, 4.8, 4.5, 4.0,
                    3.0, 2.0, 1.0, 0.5, 0.01,
                ],
                |s| list(s),
            );
            println!("tol,before_end,p_zeta,f_nl");
            for tol in list(&args[3]) {
                let mut transport = Transport::new(&system, trajectory, k, tol);
                for &before in &slices {
                    transport.advance(end - before);
                    let (f_nl, p) = transport.f_nl();
                    println!("{tol:e},{before},{p:.6e},{f_nl:.6e}");
                }
            }
        }
        "delta-n" => {
            let tol: f64 = args[3].parse().unwrap();
            let target_before: f64 = args[4].parse().unwrap();
            let slices = list(&args[5]);
            let scale = list(&args[6]);
            let scale = [scale[0], scale[1], scale[2]];
            let target_hubble = system
                .hubble(&trajectory.state_at(end - target_before))
                .into_value();
            let mut transport = Transport::new(&system, trajectory, k, tol);
            println!("n_c,f_nl,f_nl_gaussian,p_zeta");
            for before in slices {
                transport.advance(end - before);
                let (f_nl, gaussian, p) =
                    phase_space_delta_n(&transport, &settings, target_hubble, scale);
                println!("{before},{f_nl:.8e},{gaussian:.8e},{p:.8e}");
            }
            transport.advance(end - target_before);
            let (f_nl, p) = transport.f_nl();
            println!("transport,{f_nl:.8e},,{p:.8e}");
        }
        "response" => {
            // The split of `phys_type::split`: the transport up to each slice
            // and the backward response from the surface `target`.
            let tol: f64 = args[3].parse().unwrap();
            let target_before: f64 = args[4].parse().unwrap();
            let slices = list(&args[5]);
            let step: f64 = args[6].parse().unwrap();
            let target = end - target_before;
            let triangle = MomentumTriangle::try_new(k, k, k).unwrap();
            let split = SplitSettings {
                relative_tolerance: tol,
                response_step: step,
                ..SplitSettings::default()
            };
            let results = system
                .split_curvature_correlators(
                    trajectory,
                    triangle,
                    &slices.iter().map(|before| end - before).collect::<Vec<_>>(),
                    target,
                    &split,
                )
                .unwrap();
            println!("n_c,f_nl,p_zeta");
            for (before, correlators) in slices.iter().zip(&results) {
                println!(
                    "{before},{:.8e},{:.8e}",
                    correlators.reduced_f_nl(),
                    correlators.dimensionless_power()[0]
                );
            }
            let mut transport = Transport::new(&system, trajectory, k, tol);
            transport.advance(target);
            let (f_nl, p) = transport.f_nl();
            println!("transport,{f_nl:.8e},{p:.8e}");
        }
        "gap" => {
            // From the crossing-time delta N of KMS Sec. V to the split, one
            // ingredient at a time. Needs KMS_BACKGROUND_TOLERANCE=1e-13.
            let target_before: f64 = args[3].parse().unwrap();
            let deltas = list(&args[4]);
            let target = end - target_before;
            let target_hubble = system.hubble(&trajectory.state_at(target)).into_value();
            let precise = SpectrumRunSettings {
                background_sample_every: usize::MAX,
                ..settings
            };
            let star = end - 60.0;
            let x0 = [pivot.phi[0].into_value(), pivot.phi[1].into_value()];

            // 1. KMS: slow-roll velocities, N to epsilon_H = 1.
            let library = crossing_delta_n(&system, &survey, &settings, 60.0).unwrap();
            let (kms, _) = slow_roll_delta_n(&system, x0, &precise, |run| run.e_folds_at_end);
            // 2. The same with N to the surface of uniform H through the end
            // of the reference trajectory (0.001 e-folds before it; perturbed
            // trajectories that end earlier never reach the end value), and
            // through `target`.
            let to_hubble = |hubble: f64| {
                slow_roll_delta_n(&system, x0, &precise, |run| {
                    run.trajectory
                        .crossing(|st| system.hubble(st).into_value() - hubble)
                        .unwrap()
                })
                .0
            };
            let end_hubble = system
                .hubble(&trajectory.state_at(end - 1.0e-3))
                .into_value();
            println!("step,f_nl");
            println!("kms_library,{:.8e}", library.f_nl);
            println!("slow_roll_to_epsilon_1,{kms:.8e}");
            println!("slow_roll_to_uniform_h_0.001,{:.8e}", to_hubble(end_hubble));
            println!("slow_roll_to_uniform_h,{:.8e}", to_hubble(target_hubble));

            // Background at the crossing: slow-roll against the actual velocity.
            let slow = system.slow_roll_velocity(&pivot);
            let metric = system.geometry.metric(&pivot.phi);
            let speed = |v: [f64; 2]| {
                (0..4)
                    .map(|m| metric.components()[m / 2][m % 2].into_value() * v[m / 2] * v[m % 2])
                    .sum::<f64>()
                    .sqrt()
            };
            let actual = pivot.pi.map(FieldVelocity::into_value);
            let slow = slow.map(FieldVelocity::into_value);
            let curvature =
                phys_type_models::kms::KmsTheory::benchmark(1.0e-2, Formulation::Metric)
                    .gaussian_curvature(x0[0], x0[1]);
            println!(
                "# crossing: K {curvature:.6e}, epsilon_H {:.6e}, |pi| {:.8e}, slow-roll |pi| {:.8e}, angle {:.3e}",
                system.epsilon_h(&pivot),
                speed(actual),
                speed(slow),
                (actual[1].atan2(actual[0]) - slow[1].atan2(slow[0])),
            );

            // 3. The exact response of zeta on the H surface, far outside the
            // horizon (k -> 0), with massless statistics on each slice.
            let slices: Vec<f64> = deltas.iter().map(|delta| star + delta).collect();
            let split = SplitSettings {
                response_step: 1.25e-4,
                ..SplitSettings::default()
            };
            let tiny = MomentumTriangle::try_new(1.0e-6 * k, 1.0e-6 * k, 1.0e-6 * k).unwrap();
            let separate = system
                .curvature_response(trajectory, tiny, &slices, target, &split)
                .unwrap();
            let none = |_: usize, _: usize, _: usize| 0.0;
            let fields = |a: usize| a < 2;
            for (delta, (slice, response)) in deltas.iter().zip(slices.iter().zip(&separate)) {
                let inverse = system
                    .geometry
                    .inverse_metric(&trajectory.state_at(*slice).phi);
                let massless = |a: usize, b: usize| {
                    if a < 2 && b < 2 {
                        inverse.components()[a][b].into_value()
                    } else {
                        0.0
                    }
                };
                let (f_nl, _, _) = contract(response, &massless, &none, &fields);
                println!("massless_at_exit_plus_{delta},{f_nl:.8e}");
            }

            // 4. The transport up to each slice, with the response of the
            // pivot itself: fields only, with momenta, with the intrinsic part.
            let triangle = MomentumTriangle::try_new(k, k, k).unwrap();
            let responses = system
                .curvature_response(trajectory, triangle, &slices, target, &split)
                .unwrap();
            let mut transport = Transport::new(&system, trajectory, k, 1.0e-12);
            for (delta, (slice, response)) in deltas.iter().zip(slices.iter().zip(&responses)) {
                transport.advance(*slice);
                let state = &transport.state;
                let sigma = |a: usize, b: usize| covariance_entry(&state.covariances[0], a, b);
                let alpha = |a: usize, b: usize, c: usize| {
                    let (i, j, l) = (a % 2, b % 2, c % 2);
                    let block = &state.alpha;
                    match (a < 2, b < 2, c < 2) {
                        (true, true, true) => block.qqq.components()[i][j][l].value().re,
                        (true, true, false) => block.qqp.components()[i][j][l].value().re,
                        (true, false, true) => block.qpq.components()[i][j][l].value().re,
                        (false, true, true) => block.pqq.components()[i][j][l].value().re,
                        (true, false, false) => block.qpp.components()[i][j][l].value().re,
                        (false, true, false) => block.pqp.components()[i][j][l].value().re,
                        (false, false, true) => block.ppq.components()[i][j][l].value().re,
                        (false, false, false) => block.ppp.components()[i][j][l].value().re,
                    }
                };
                let all = |_: usize| true;
                let (q_only, _, _) = contract(response, &sigma, &none, &fields);
                let (gaussian, _, _) = contract(response, &sigma, &none, &all);
                let (full, _, intrinsic) = contract(response, &sigma, &alpha, &all);
                println!("transport_fields_at_exit_plus_{delta},{q_only:.8e}");
                println!("transport_gaussian_at_exit_plus_{delta},{gaussian:.8e}");
                println!(
                    "transport_full_at_exit_plus_{delta},{full:.8e},intrinsic {intrinsic:.8e}"
                );
            }
        }
        "noise" => {
            // N along chi from the slice `n_c`, to the H surface and to epsilon_H = 1.
            let n_c: f64 = args[3].parse().unwrap();
            let target_before: f64 = args[4].parse().unwrap();
            let base = trajectory.state_at(end - n_c);
            let target_hubble = system
                .hubble(&trajectory.state_at(end - target_before))
                .into_value();
            let precise = SpectrumRunSettings {
                tolerances: ErrorTolerances::new(
                    ConformalPerturbationTolerance::<2>::background_in_planck_units(1.0e-24),
                    1.0e-13,
                ),
                background_sample_every: usize::MAX,
                ..settings
            };
            let h = 1.0e-4 * base.phi[1].into_value().abs();
            println!("t,n_h,n_end");
            for step in -8..=8 {
                let t = f64::from(step) / 8.0;
                let mut start = base;
                start.phi[1] = Field::new(base.phi[1].into_value() + t * h);
                start.a = ScaleFactor::new(1.0);
                let run = system.survey(start, &precise).unwrap();
                let n_h = run
                    .trajectory
                    .crossing(|st| system.hubble(st).into_value() - target_hubble)
                    .unwrap();
                println!("{t},{n_h:.15e},{:.15e}", run.e_folds_at_end);
            }
        }
        _ => panic!("unknown mode"),
    }
}
