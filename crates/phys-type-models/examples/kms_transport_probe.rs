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
//! - `spectra <index> <before,...>`: P_R, P_S, and n_s of the pivot read off
//!   `before` e-folds before the end (G3), then the slow-roll n_s(t_*) of
//!   KMS Eq. (85) and T_RS from t_* by KMS Eqs. (70), (72), and (77).
//!   `KMS_TILT_STEP` sets the step in ln k of the central difference.
//! - `after <index> <after> <t,...>`: the split from slices 20 and 10
//!   e-folds before the end, read off `t` e-folds after the end of inflation
//!   (negative before it), with the background continued `after` e-folds
//!   past the end.
//! - `masses <index> <after> <t0> <t1> <dt>`: the mass of the isocurvature
//!   direction in units of `H^2` from `t0` to `t1` e-folds after the end:
//!   the potential, the Riemann term, and the turn.
//! - `orbit <index> <after> <t0> <t1> <dt> [<t_s> <scale>]`: the background
//!   from `t0` to `t1` e-folds after the end. With `t_s` and `scale`, the
//!   background restarts at `t_s` e-folds after the end with `phi` and its
//!   velocity multiplied by `scale`, and `t` counts from the same end.
//! - `floquet <index> <xi_scale> <A,...>`: without expansion, the growth of
//!   a small `phi` per half period of `chi` oscillating with amplitude `A`
//!   (Floquet). `xi_scale` multiplies both nonminimal couplings; 0 is the
//!   canonical limit. `<index>` is not used.
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
    Dimensionless, ErrorTolerances, Field, FieldSpaceGeometry, FieldVelocity, Formulation, Mass,
    MomentumTriangle, MultifieldState, ScaleFactor, SpectrumRunSettings, SplitSettings,
    StepSizeLimits, adaptive_step, canonical_three_point_transport_initial_state,
    curvature_correlators, next_step_size, velocity_three_point_from_transport_state,
};
use phys_type_models::kms::{
    KmsSystem, PHI_START, TRAJECTORIES, at_rest, benchmark_system, crossing_delta_n,
    slow_roll_tilt, transfer_functions,
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

/// `d/dt (phi^I, phidot^I)` without expansion: the geodesic equation with
/// the force of the potential, in Planck units.
fn flat_rate(system: &KmsSystem, y: [f64; 4]) -> [f64; 4] {
    let position = [Field::new(y[0]), Field::new(y[1])];
    let connection = system.geometry.christoffel(&position);
    let inverse = system.geometry.inverse_metric(&position);
    let gradient = phys_type::MultifieldPotential::gradient(&system.potential, position);
    let mut out = [y[2], y[3], 0.0, 0.0];
    for i in 0..2 {
        let geodesic: f64 = (0..4)
            .map(|m| connection.components()[i][m / 2][m % 2].value() * y[2 + m / 2] * y[2 + m % 2])
            .sum();
        let force: f64 = (0..2)
            .map(|j| inverse.components()[i][j].into_value() * gradient[j].into_value())
            .sum();
        out[2 + i] = -geodesic - force;
    }
    out
}

/// RK4 from `y` over `time` in `steps` equal steps.
fn flat_evolve(system: &KmsSystem, mut y: [f64; 4], time: f64, steps: usize) -> [f64; 4] {
    let h = time / steps as f64;
    let add = |y: [f64; 4], k: [f64; 4], c: f64| -> [f64; 4] {
        core::array::from_fn(|i| y[i] + c * k[i])
    };
    for _ in 0..steps {
        let k1 = flat_rate(system, y);
        let k2 = flat_rate(system, add(y, k1, 0.5 * h));
        let k3 = flat_rate(system, add(y, k2, 0.5 * h));
        let k4 = flat_rate(system, add(y, k3, h));
        y = core::array::from_fn(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    }
    y
}

/// (period of chi, growth of phi per half period) at chi amplitude `A`.
fn floquet(system: &KmsSystem, amplitude: f64, steps: usize) -> (f64, f64) {
    let start = [0.0, amplitude, 0.0, 0.0];
    // Bracket the return to the apex (chidot from + to -), then secant.
    let scale = 1.0 / (0.005_f64.sqrt() * amplitude);
    let dt = scale / 2000.0;
    let (mut y, mut t, mut previous) = (start, 0.0, 0.0);
    loop {
        y = flat_evolve(system, y, dt, 1);
        t += dt;
        if previous > 0.0 && y[3] <= 0.0 {
            break;
        }
        previous = y[3];
    }
    let speed_at = |time: f64| flat_evolve(system, start, time, steps)[3];
    let (mut a, mut b) = (t - dt, t);
    let (mut fa, mut fb) = (speed_at(a), speed_at(b));
    for _ in 0..30 {
        let c = b - fb * (b - a) / (fb - fa);
        let fc = speed_at(c);
        (a, fa, b, fb) = (b, fb, c, fc);
        if (b - a).abs() < 1.0e-13 * b {
            break;
        }
    }
    let period = b;
    let seed = 1.0e-10;
    let unit = 0.005_f64.sqrt() * amplitude;
    let q = flat_evolve(system, [seed, amplitude, 0.0, 0.0], period, steps);
    let p = flat_evolve(system, [0.0, amplitude, seed * unit, 0.0], period, steps);
    // Monodromy on (phi, phidot / unit).
    let m = [
        [q[0] / seed, p[0] / seed],
        [q[2] / (seed * unit), p[2] / (seed * unit)],
    ];
    let trace = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let disc = trace * trace - 4.0 * det;
    let largest = if disc >= 0.0 {
        (0.5 * (trace.abs() + disc.sqrt())).abs()
    } else {
        det.abs().sqrt()
    };
    (period, largest.sqrt())
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
        // KMS_EXIT_DEPTH overrides how many e-folds before horizon exit the
        // modes start in the Bunch-Davies vacuum.
        e_folds_before_exit: std::env::var("KMS_EXIT_DEPTH").map_or(6.0, |v| v.parse().unwrap()),
        background_sample_every: 1,
        tolerances: ErrorTolerances::new(
            ConformalPerturbationTolerance::<2>::background_in_planck_units(
                1.0e-10 * background_tolerance,
            ),
            background_tolerance,
        ),
        ..SpectrumRunSettings::default()
    };
    let settings = if mode == "after" || mode == "masses" || mode == "orbit" {
        SpectrumRunSettings {
            e_folds_after_end: args[3].parse().unwrap(),
            ..settings
        }
    } else {
        settings
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
        "after" => {
            // f_NL read off after the end of inflation. Needs
            // KMS_BACKGROUND_TOLERANCE=1e-13.
            let triangle = MomentumTriangle::try_new(k, k, k).unwrap();
            // KMS_RESPONSE_STEP overrides the RK4 step of the response.
            let split = SplitSettings {
                response_step: std::env::var("KMS_RESPONSE_STEP")
                    .map_or(2.5e-4, |value| value.parse().unwrap()),
                ..SplitSettings::default()
            };
            println!("t,epsilon_h,f_nl_20,f_nl_10,p_zeta");
            for t in list(&args[4]) {
                let output = end + t;
                let results = system
                    .split_curvature_correlators(
                        trajectory,
                        triangle,
                        &[end - 20.0, end - 10.0],
                        output,
                        &split,
                    )
                    .unwrap();
                println!(
                    "{t},{:.6e},{:.8e},{:.8e},{:.8e}",
                    system.epsilon_h(&trajectory.state_at(output)),
                    results[0].reduced_f_nl(),
                    results[1].reduced_f_nl(),
                    results[0].dimensionless_power()[0]
                );
            }
        }
        "masses" => {
            // C^I_J projected on the unit isocurvature direction e_s, split
            // into the covariant Hessian of the potential and the Riemann
            // term, with the turn eta_perp^2 that enters the superhorizon
            // isocurvature mass mu_s^2 = C_ss - omega^2.
            let (t0, t1, dt): (f64, f64, f64) = (
                args[4].parse().unwrap(),
                args[5].parse().unwrap(),
                args[6].parse().unwrap(),
            );
            let theory = phys_type_models::kms::KmsTheory::benchmark(1.0e-2, Formulation::Metric);
            println!("t,epsilon_h,K,hessian_ss,riemann_ss,eta_perp2,mu_s2,phi,chi");
            let count = ((t1 - t0) / dt).round() as usize;
            for step in 0..=count {
                let t = t0 + dt * step as f64;
                let state = trajectory.state_at(end + t);
                let Ok(kinematics) = system.kinematics(&state) else {
                    continue;
                };
                let metric = system.geometry.metric(&state.phi);
                let g = |i: usize, j: usize| metric.components()[i][j].into_value();
                let e: [f64; 2] =
                    core::array::from_fn(|i| kinematics.tangent.components()[i].into_value());
                // e_s: rotate e_sigma by 90 degrees in the metric.
                let lowered: [f64; 2] =
                    core::array::from_fn(|i| (0..2).map(|j| g(i, j) * e[j]).sum());
                let det = g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0);
                let es = [-lowered[1] / det.sqrt(), lowered[0] / det.sqrt()];
                let es_low: [f64; 2] =
                    core::array::from_fn(|i| (0..2).map(|j| g(i, j) * es[j]).sum());
                let hubble2 = system.hubble(&state).into_value().powi(2);
                let hessian = system.covariant_potential_hessian(&state);
                let hessian_ss: f64 = (0..4)
                    .map(|m| es[m / 2] * es[m % 2] * hessian.components()[m / 2][m % 2].value())
                    .sum::<f64>()
                    / hubble2;
                let total = system.perturbation_mass_matrix(&state);
                let total_ss: f64 = (0..4)
                    .map(|m| es_low[m / 2] * total.components()[m / 2][m % 2].value() * es[m % 2])
                    .sum::<f64>()
                    / hubble2;
                // The gravitational part of C is proportional to the velocity
                // on its right, so it drops out along e_s.
                let riemann_ss = total_ss - hessian_ss;
                let eta2 = kinematics.eta_perp.into_value().powi(2);
                let [phi, chi] = state.phi.map(|f| f.into_value());
                println!(
                    "{t:.4},{:.4e},{:.4e},{hessian_ss:.4e},{riemann_ss:.4e},{eta2:.4e},{:.4e},{phi:.4e},{chi:.4e}",
                    system.epsilon_h(&state),
                    theory.gaussian_curvature(phi, chi),
                    total_ss - eta2,
                );
            }
        }
        "orbit" => {
            let (t0, t1, dt): (f64, f64, f64) = (
                args[4].parse().unwrap(),
                args[5].parse().unwrap(),
                args[6].parse().unwrap(),
            );
            let after: f64 = args[3].parse().unwrap();
            let restarted = args.get(7).map(|t_s| {
                let t_s: f64 = t_s.parse().unwrap();
                let scale: f64 = args[8].parse().unwrap();
                let mut start = trajectory.state_at(end + t_s);
                start.phi[0] = Field::new(scale * start.phi[0].into_value());
                start.pi[0] = FieldVelocity::new(scale * start.pi[0].into_value());
                start.a = ScaleFactor::new(1.0);
                // Inflation has ended; the survey stops at once (epsilon_H
                // > 1) or within a few steps, then continues.
                let run = system
                    .survey(
                        start,
                        &SpectrumRunSettings {
                            e_folds_after_end: after - t_s,
                            ..settings
                        },
                    )
                    .unwrap();
                (run, t_s)
            });
            println!("t,phi,chi,pi_phi,pi_chi,epsilon_h,hubble");
            let count = ((t1 - t0) / dt).round() as usize;
            for step in 0..=count {
                let t = t0 + dt * step as f64;
                let state = match &restarted {
                    None => trajectory.state_at(end + t),
                    Some((run, t_s)) => {
                        if t < *t_s {
                            continue;
                        }
                        run.trajectory.state_at(t - t_s)
                    }
                };
                let [phi, chi] = state.phi.map(|f| f.into_value());
                let [pi_phi, pi_chi] = state.pi.map(|v| v.into_value());
                println!(
                    "{t:.5},{phi:.6e},{chi:.6e},{pi_phi:.6e},{pi_chi:.6e},{:.6e},{:.6e}",
                    system.epsilon_h(&state),
                    system.hubble(&state).into_value()
                );
            }
        }
        "floquet" => {
            let xi_scale: f64 = args[3].parse().unwrap();
            let base = phys_type_models::kms::KmsTheory::benchmark(1.0e-2, Formulation::Metric);
            let theory = phys_type_models::kms::KmsTheory {
                xi_phi: xi_scale * base.xi_phi,
                xi_chi: xi_scale * base.xi_chi,
                ..base
            };
            let (potential, geometry) = phys_type::einstein_frame(theory);
            let scaled = phys_type::CovariantMultifieldSystem::in_planck_units(potential, geometry);
            // KMS_FLOQUET_STEPS sets the RK4 steps per period.
            let steps = std::env::var("KMS_FLOQUET_STEPS").map_or(20_000, |v| v.parse().unwrap());
            println!("amplitude,period,growth_per_half_period");
            for amplitude in list(&args[4]) {
                let (period, growth) = floquet(&scaled, amplitude, steps);
                println!("{amplitude},{period:.8e},{growth:.6}");
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
        "spectra" => {
            // G3: P_R and n_s of the pivot (N_* = 60) read off on each slice,
            // n_s from a central difference in ln k; then the slow-roll n_s(t_*)
            // of KMS Eq. (85) and T_RS from t_* to `before` by KMS Eqs. (70),
            // (72), and (77) on the exact background.
            let step: f64 = std::env::var("KMS_TILT_STEP").map_or(0.02, |v| v.parse().unwrap());
            println!("before_end,p_r,p_s,correlation,n_s");
            for before in list(&args[3]) {
                let output = end - before;
                let spectra = |kk: f64| {
                    system
                        .mode_spectrum(&settings, trajectory, Mass::new(kk), output)
                        .unwrap()
                        .spectra
                };
                let (low, central, high) = (
                    spectra(k * (-step).exp()),
                    spectra(k),
                    spectra(k * step.exp()),
                );
                let n_s = 1.0
                    + (high.curvature.into_value() / low.curvature.into_value()).ln()
                        / (2.0 * step);
                println!(
                    "{before},{:.8e},{:.8e},{:.6},{n_s:.6}",
                    central.curvature.into_value(),
                    central.entropy.into_value(),
                    central.correlation.unwrap_or(f64::NAN),
                );
            }
            let tilt = slow_roll_tilt(&system, &survey, 60.0).unwrap();
            println!("# n_s(t_*) = 1 - 6 epsilon + 2 eta_sigma_sigma = {tilt:.6}");
            let dn: f64 = std::env::var("KMS_TRS_STEP").map_or(1.0e-3, |v| v.parse().unwrap());
            println!("# T_RS by KMS Eqs. (70), (72), (77)");
            println!("before_end,t_rs,t_ss");
            for before in list(&args[3]).into_iter().filter(|&b| b < 60.0) {
                let (t_rs, t_ss) = transfer_functions(&system, &survey, 60.0, before, dn).unwrap();
                println!("{before},{t_rs:.6e},{t_ss:.6e}");
            }
        }
        _ => panic!("unknown mode"),
    }
}
