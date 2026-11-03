//! Tree-level `zeta` correlators split at a superhorizon slice.
//!
//! Where the correlators grow by orders of magnitude and then cancel, as in
//! the valley after a sharp turn, the transport carried to the end of
//! inflation keeps a relative error of order the tolerance times the
//! growth. [`CovariantMultifieldSystem::split_curvature_correlators`] stops
//! the transport on a slice `t_c` before the growth and continues with the
//! response of the final `zeta` to the phase space on that slice,
//!
//! `zeta_e = n_a X^a(t_c) + (1/2) n_ab X^a(t_c) X^b(t_c)`.
//!
//! The response obeys the adjoint of the transport's own flows,
//! `D_N n_a = -n_b u^b_a` and
//! `D_N n_ab = -n_c u^c_ab - n_cb u^c_a - n_ac u^c_b`, from the gauge
//! coefficients `N_a`, `N_ab` on the output slice, integrated backwards with
//! RK4. The split is exact up to `(k/aH)^2` on the slice, so the result does
//! not depend on `t_c`; comparing two slices is the check. See appendix F of
//! `notes/overview`.

use core::fmt;

use crate::bispectrum_run::{ThreePointEvolution, ThreePointEvolutionError};
use crate::curvature::{
    CurvatureCorrelators, CurvatureProjectionError, curvature_gauge_coefficients,
};
use crate::spectrum_run::BackgroundTrajectory;
use crate::transport::{
    CanonicalKernelError, CanonicalQuadraticPhaseSpaceKernel, CanonicalThreePointTransportState,
    CubicCoefficientError, MomentumTriangle, ThreePointBlock, canonical_cubic_hamiltonian_vertices,
    linear_phase_space_flows,
};
use crate::{
    CovariantMultifieldSystem, CurvatureDerivativeGeometry, MMinus3, MMinus6, Mass, Quantity,
    ThirdOrderFieldSpaceGeometry, ThriceDifferentiableMultifieldPotential,
};

/// Numerical settings of [`CovariantMultifieldSystem::split_curvature_correlators`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitSettings {
    /// The transport starts where the shortest leg has
    /// `k/(aH) = exp(e_folds_before_exit)`.
    pub e_folds_before_exit: f64,
    /// Relative tolerance of the transport up to the slice.
    pub relative_tolerance: f64,
    /// Attempted transport steps allowed up to each slice.
    pub max_steps: usize,
    /// Largest RK4 step of the response, in e-folds. Steps never cross a
    /// background checkpoint, where the interpolated background is only
    /// `C^1`.
    pub response_step: f64,
    /// Every leg must have `k/(aH)` at most this on the output slice.
    pub maximum_output_k_over_ah: f64,
}

impl Default for SplitSettings {
    fn default() -> Self {
        Self {
            e_folds_before_exit: 5.0,
            relative_tolerance: 1.0e-12,
            max_steps: 5_000_000,
            response_step: 5.0e-4,
            maximum_output_k_over_ah: 0.1,
        }
    }
}

/// Failure of [`CovariantMultifieldSystem::split_curvature_correlators`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitError {
    /// A setting or slice is out of range.
    InvalidSettings(&'static str),
    /// The transport up to a slice failed.
    Transport(ThreePointEvolutionError),
    /// The gauge coefficients on the output slice could not be formed.
    Projection(CurvatureProjectionError),
    /// The cubic vertices could not be formed on the background.
    Vertices(CubicCoefficientError),
    /// The quadratic flow could not be formed on the background.
    Kernel(CanonicalKernelError),
}

impl fmt::Display for SplitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSettings(message) => formatter.write_str(message),
            Self::Transport(source) => write!(formatter, "transport up to the slice: {source}"),
            Self::Projection(source) => write!(formatter, "output projection: {source:?}"),
            Self::Vertices(source) => write!(formatter, "cubic vertices: {source:?}"),
            Self::Kernel(source) => write!(formatter, "quadratic flow: {source}"),
        }
    }
}

impl std::error::Error for SplitError {}

/// The response of the final `zeta` to the phase space `(Q^I, P^I)` on a
/// slice: index `a < N` is `Q^a`, `a >= N` is `P^(a-N)`. `linear[r]` is
/// `n_a` of leg `r`; `quadratic[r]` is `n^(r)_ab` with output leg `r` and
/// inputs in ascending leg order, flattened row-major. With every leg far
/// outside the horizon these are the phase-space delta N coefficients
/// `N_a`, `N_ab` in the transport's variables.
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureResponse {
    /// `n_a` of each leg, length `2N`.
    pub linear: [Vec<f64>; 3],
    /// `n^(r)_ab` of each output leg, row-major `2N x 2N`.
    pub quadratic: [Vec<f64>; 3],
}

type Response = CurvatureResponse;

impl Response {
    fn axpy(&self, scale: f64, other: &Self) -> Self {
        let add = |x: &Vec<f64>, y: &Vec<f64>| -> Vec<f64> {
            x.iter().zip(y).map(|(x, y)| x + scale * y).collect()
        };
        Self {
            linear: core::array::from_fn(|r| add(&self.linear[r], &other.linear[r])),
            quadratic: core::array::from_fn(|r| add(&self.quadratic[r], &other.quadratic[r])),
        }
    }
}

/// The linear flow of each leg and the quadratic flow of each output leg on
/// the background at `e_folds`, as plain arrays on the `2N` phase space.
struct Flows {
    /// `u^a_b` of leg `r`, row-major `2N x 2N`.
    linear: [Vec<f64>; 3],
    /// `u^(r)a_bc`, row-major `2N x 2N x 2N`.
    quadratic: [Vec<f64>; 3],
}

fn kernel_entries<const N: usize>(kernel: &CanonicalQuadraticPhaseSpaceKernel<N>) -> Vec<f64> {
    fn block<A, B, C, D, const N: usize>(
        block: &ThreePointBlock<A, B, C, D, f64, N>,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        *block.components()[i][j][k].value()
    }
    let m = 2 * N;
    let mut out = vec![0.0; m * m * m];
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let (i, j, k) = (a % N, b % N, c % N);
                out[(a * m + b) * m + c] = match (a < N, b < N, c < N) {
                    (true, true, true) => block(&kernel.q_qq, i, j, k),
                    (true, true, false) => block(&kernel.q_qp, i, j, k),
                    (true, false, true) => block(&kernel.q_pq, i, j, k),
                    (true, false, false) => block(&kernel.q_pp, i, j, k),
                    (false, true, true) => block(&kernel.p_qq, i, j, k),
                    (false, true, false) => block(&kernel.p_qp, i, j, k),
                    (false, false, true) => block(&kernel.p_pq, i, j, k),
                    (false, false, false) => block(&kernel.p_pp, i, j, k),
                };
            }
        }
    }
    out
}

fn flows<P, G, const N: usize>(
    system: &CovariantMultifieldSystem<P, G, N>,
    trajectory: &BackgroundTrajectory<N>,
    triangle: MomentumTriangle,
    e_folds: f64,
) -> Result<Flows, SplitError>
where
    P: ThriceDifferentiableMultifieldPotential<N>,
    G: ThirdOrderFieldSpaceGeometry<N> + CurvatureDerivativeGeometry<N>,
{
    let background = trajectory.state_at(e_folds);
    let wavenumbers = triangle.canonical_wavenumbers().map(Mass::new);
    let m = 2 * N;
    let linear = linear_phase_space_flows(system, &background, wavenumbers).map(|flow| {
        let mut out = vec![0.0; m * m];
        for i in 0..N {
            for j in 0..N {
                out[i * m + j] = flow.qq[i][j];
                out[i * m + N + j] = flow.qp[i][j];
                out[(N + i) * m + j] = flow.pq[i][j];
                out[(N + i) * m + N + j] = flow.pp[i][j];
            }
        }
        out
    });
    let kernels = canonical_cubic_hamiltonian_vertices(system, &background, triangle)
        .map_err(SplitError::Vertices)?
        .e_fold_kernels(system.hubble(&background))
        .map_err(SplitError::Kernel)?;
    Ok(Flows {
        linear,
        quadratic: [
            kernel_entries(&kernels[0]),
            kernel_entries(&kernels[1]),
            kernel_entries(&kernels[2]),
        ],
    })
}

/// The inputs of output leg `r`, in ascending order.
const INPUTS: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];

/// `d n / dN` from the adjoint of `D_N X = u X + (1/2) u_bc X X`.
fn rate(response: &Response, flows: &Flows, m: usize) -> Response {
    let linear = core::array::from_fn(|r| {
        let (n, u) = (&response.linear[r], &flows.linear[r]);
        (0..m)
            .map(|a| -(0..m).map(|b| n[b] * u[b * m + a]).sum::<f64>())
            .collect()
    });
    let quadratic = core::array::from_fn(|r| {
        let [s, t] = INPUTS[r];
        let (n, q) = (&response.linear[r], &response.quadratic[r]);
        let (kernel, us, ut) = (&flows.quadratic[r], &flows.linear[s], &flows.linear[t]);
        let mut out = vec![0.0; m * m];
        for a in 0..m {
            for b in 0..m {
                out[a * m + b] = -(0..m)
                    .map(|c| {
                        n[c] * kernel[(c * m + a) * m + b]
                            + q[c * m + b] * us[c * m + a]
                            + q[a * m + c] * ut[c * m + b]
                    })
                    .sum::<f64>();
            }
        }
        out
    });
    Response { linear, quadratic }
}

/// Real part of the two-point entry `(a, b)` of a canonical covariance.
fn covariance_entry<const N: usize>(
    state: &CanonicalThreePointTransportState<N>,
    leg: usize,
    a: usize,
    b: usize,
) -> f64 {
    let covariance = &state.covariances[leg];
    let (i, j) = (a % N, b % N);
    match (a < N, b < N) {
        (true, true) => covariance.qq.components()[i][j].value().re,
        (true, false) => covariance.qp.components()[i][j].value().re,
        (false, true) => covariance.pq.components()[i][j].value().re,
        (false, false) => covariance.pp.components()[i][j].value().re,
    }
}

/// Real part of the three-point entry `(a, b, c)` of the canonical state.
fn alpha_entry<const N: usize>(
    state: &CanonicalThreePointTransportState<N>,
    a: usize,
    b: usize,
    c: usize,
) -> f64 {
    let alpha = &state.alpha;
    let (i, j, k) = (a % N, b % N, c % N);
    match (a < N, b < N, c < N) {
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

/// `P_zeta` and `B_zeta` from the response and the transport on one slice.
fn combine<const N: usize>(
    response: &Response,
    state: &CanonicalThreePointTransportState<N>,
    triangle: MomentumTriangle,
) -> CurvatureCorrelators {
    let m = 2 * N;
    let power: [f64; 3] = core::array::from_fn(|leg| {
        let n = &response.linear[leg];
        (0..m * m)
            .map(|x| n[x / m] * n[x % m] * covariance_entry(state, leg, x / m, x % m))
            .sum()
    });
    let projected: [Vec<f64>; 3] = core::array::from_fn(|leg| {
        let n = &response.linear[leg];
        (0..m)
            .map(|a| {
                (0..m)
                    .map(|c| n[c] * covariance_entry(state, leg, c, a))
                    .sum()
            })
            .collect()
    });
    let [n1, n2, n3] = &response.linear;
    let intrinsic: f64 = (0..m * m * m)
        .map(|x| {
            let [a, b, c] = [x / (m * m), (x / m) % m, x % m];
            n1[a] * n2[b] * n3[c] * alpha_entry(state, a, b, c)
        })
        .sum();
    let gauge: f64 = (0..3)
        .map(|r| {
            let [s, t] = INPUTS[r];
            (0..m * m)
                .map(|x| response.quadratic[r][x] * projected[s][x / m] * projected[t][x % m])
                .sum::<f64>()
        })
        .sum();
    CurvatureCorrelators {
        wavenumbers: triangle.canonical_wavenumbers().map(Mass::new),
        power: power.map(Quantity::<MMinus3, f64>::new),
        bispectrum: Quantity::<MMinus6, f64>::new(intrinsic + gauge),
        bispectrum_imaginary_ratio: 0.0,
        intrinsic: Quantity::<MMinus6, f64>::new(intrinsic),
    }
}

impl<P, G, const N: usize> CovariantMultifieldSystem<P, G, N>
where
    P: ThriceDifferentiableMultifieldPotential<N> + Copy,
    G: ThirdOrderFieldSpaceGeometry<N> + CurvatureDerivativeGeometry<N> + Copy,
{
    /// The response of `zeta` on the output slice `output_e_folds` to the
    /// phase space on each slice of `slices` (elapsed e-folds, before
    /// `output_e_folds`), integrated backwards from the gauge coefficients.
    /// Results come in the order of `slices`.
    pub fn curvature_response(
        &self,
        trajectory: &BackgroundTrajectory<N>,
        triangle: MomentumTriangle,
        slices: &[f64],
        output_e_folds: f64,
        settings: &SplitSettings,
    ) -> Result<Vec<CurvatureResponse>, SplitError> {
        if !(settings.response_step > 0.0 && settings.response_step.is_finite()) {
            return Err(SplitError::InvalidSettings(
                "response_step must be positive and finite",
            ));
        }
        if slices
            .iter()
            .any(|slice| !slice.is_finite() || *slice >= output_e_folds)
        {
            return Err(SplitError::InvalidSettings(
                "every slice must lie before the output slice",
            ));
        }
        let m = 2 * N;

        // The response on the output slice: the gauge coefficients.
        let output = trajectory.state_at(output_e_folds);
        let gauge = curvature_gauge_coefficients(
            self,
            &output,
            triangle,
            settings.maximum_output_k_over_ah,
        )
        .map_err(SplitError::Projection)?;
        let mut linear = vec![0.0; m];
        for (i, value) in gauge.linear.iter().enumerate() {
            linear[i] = value.into_value();
        }
        let mut response = Response {
            linear: core::array::from_fn(|_| linear.clone()),
            quadratic: gauge.quadratic.map(|vertex| {
                let mut out = vec![0.0; m * m];
                for i in 0..N {
                    for j in 0..N {
                        out[i * m + j] = vertex.field_field[i][j].into_value();
                        out[i * m + N + j] = vertex.field_momentum[i][j].into_value();
                        out[(N + i) * m + j] = vertex.momentum_field[i][j].into_value();
                    }
                }
                out
            }),
        };

        // Backwards to each slice, latest first, never stepping across a
        // checkpoint.
        let mut order: Vec<usize> = (0..slices.len()).collect();
        order.sort_by(|&x, &y| slices[y].total_cmp(&slices[x]));
        let knots: Vec<f64> = trajectory.checkpoint_e_folds().collect();
        let mut responses: Vec<Option<Response>> = vec![None; slices.len()];
        let mut e_folds = output_e_folds;
        for &index in &order {
            let stop = slices[index];
            let mut nodes: Vec<f64> = knots
                .iter()
                .copied()
                .filter(|&knot| knot < e_folds - 1.0e-12 && knot > stop + 1.0e-12)
                .rev()
                .collect();
            nodes.push(stop);
            for node in nodes {
                let width = e_folds - node;
                let count = (width / settings.response_step).ceil().max(1.0);
                let h = -width / count;
                for _ in 0..count as usize {
                    let at = |n: f64, y: &Response| -> Result<Response, SplitError> {
                        Ok(rate(y, &flows(self, trajectory, triangle, n)?, m))
                    };
                    let k1 = at(e_folds, &response)?;
                    let k2 = at(e_folds + 0.5 * h, &response.axpy(0.5 * h, &k1))?;
                    let k3 = at(e_folds + 0.5 * h, &response.axpy(0.5 * h, &k2))?;
                    let k4 = at(e_folds + h, &response.axpy(h, &k3))?;
                    response = response
                        .axpy(h / 6.0, &k1)
                        .axpy(h / 3.0, &k2)
                        .axpy(h / 3.0, &k3)
                        .axpy(h / 6.0, &k4);
                    e_folds += h;
                }
                e_folds = node;
            }
            responses[index] = Some(response.clone());
        }

        Ok(responses
            .into_iter()
            .map(|response| response.expect("every slice has a response"))
            .collect())
    }

    /// `P_zeta` and `B_zeta` of `triangle` on the output slice
    /// `output_e_folds`, split at each slice of `slices` (elapsed e-folds,
    /// superhorizon and before `output_e_folds`): the transport up to the
    /// slice, then the backward response from the output slice. Results come
    /// in the order of `slices`. `bispectrum_imaginary_ratio` is zero; only
    /// the real parts on the slice enter.
    ///
    /// The slices must lie before any large transient growth of the
    /// correlators, or the cancellation stays inside the contraction.
    pub fn split_curvature_correlators(
        &self,
        trajectory: &BackgroundTrajectory<N>,
        triangle: MomentumTriangle,
        slices: &[f64],
        output_e_folds: f64,
        settings: &SplitSettings,
    ) -> Result<Vec<CurvatureCorrelators>, SplitError> {
        if !(settings.relative_tolerance > 0.0 && settings.relative_tolerance < 1.0) {
            return Err(SplitError::InvalidSettings(
                "relative_tolerance must lie in (0, 1)",
            ));
        }
        let responses =
            self.curvature_response(trajectory, triangle, slices, output_e_folds, settings)?;
        let mut order: Vec<usize> = (0..slices.len()).collect();
        order.sort_by(|&x, &y| slices[y].total_cmp(&slices[x]));

        // Forwards with the transport, earliest slice first.
        let mut evolution =
            ThreePointEvolution::start(self, trajectory, triangle, settings.e_folds_before_exit)
                .map_err(SplitError::Transport)?;
        let mut results: Vec<Option<CurvatureCorrelators>> = vec![None; slices.len()];
        for &index in order.iter().rev() {
            if slices[index] < evolution.start {
                return Err(SplitError::InvalidSettings(
                    "every slice must lie after the subhorizon start of the transport",
                ));
            }
            evolution
                .advance(
                    slices[index],
                    settings.relative_tolerance,
                    settings.max_steps,
                )
                .map_err(SplitError::Transport)?;
            results[index] = Some(combine(&responses[index], &evolution.state, triangle));
        }
        Ok(results
            .into_iter()
            .map(|result| result.expect("every slice is evaluated"))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BispectrumRunSettings, Field, FieldVelocity, FlatFieldSpace, MultifieldState,
        QuadraticMultifieldPotential, ScaleFactor, SpectrumRunSettings, curvature_correlators,
    };

    #[test]
    fn split_reproduces_the_transport_on_any_slice() {
        // The turning double-quadratic benchmark of the bispectrum run, where
        // the transport converges to the output slice.
        let system = CovariantMultifieldSystem::in_planck_units(
            QuadraticMultifieldPotential::new([Mass::new(1.0e-6), Mass::new(1.5e-6)]),
            FlatFieldSpace,
        );
        let resting = MultifieldState::new(
            [Field::new(12.0), Field::new(8.0)],
            [FieldVelocity::new(0.0); 2],
            ScaleFactor::new(1.0),
        );
        let initial = MultifieldState {
            pi: system.slow_roll_velocity(&resting),
            ..resting
        };
        let background = SpectrumRunSettings {
            background_sample_every: usize::MAX,
            ..SpectrumRunSettings::default()
        };
        let survey = system.survey(initial, &background).unwrap();
        let end = survey.e_folds_at_end;
        let pivot = survey.trajectory.state_at(end - 6.0);
        let k = (pivot.a * system.hubble(&pivot)).into_value();
        let triangle = MomentumTriangle::try_new(k, 0.9 * k, 0.8 * k).unwrap();
        let output = end - 2.0;
        let settings = SplitSettings {
            e_folds_before_exit: 3.0,
            relative_tolerance: 1.0e-8,
            response_step: 1.0e-2,
            ..SplitSettings::default()
        };
        let transport = BispectrumRunSettings::default();
        let (_, state) = crate::bispectrum_run::evolve_three_point(
            &system,
            &survey.trajectory,
            triangle,
            settings.e_folds_before_exit,
            output,
            settings.relative_tolerance,
            transport.max_steps,
        )
        .unwrap();
        let direct = curvature_correlators(
            &system,
            &survey.trajectory.state_at(output),
            triangle,
            &state,
            settings.maximum_output_k_over_ah,
        )
        .unwrap();

        let split = system
            .split_curvature_correlators(
                &survey.trajectory,
                triangle,
                &[end - 3.0, end - 4.0, output],
                output,
                &settings,
            )
            .unwrap_err();
        assert!(matches!(split, SplitError::InvalidSettings(_)));
        let split = system
            .split_curvature_correlators(
                &survey.trajectory,
                triangle,
                &[end - 2.5, end - 3.0],
                output,
                &settings,
            )
            .unwrap();
        let reference = direct.reduced_f_nl();
        for correlators in &split {
            let f_nl = correlators.reduced_f_nl();
            assert!(
                (f_nl - reference).abs() < 1.0e-6 * reference.abs(),
                "{f_nl} vs {reference}"
            );
            for leg in 0..3 {
                let (p, q) = (correlators.power[leg], direct.power[leg]);
                assert!(((p - q) / q).into_value().abs() < 1.0e-8, "leg {leg}");
            }
        }
    }
}
