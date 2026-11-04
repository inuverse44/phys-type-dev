//! The three trajectories of arXiv:1210.7487 (KMS) off the ridge `chi = 0`:
//! their background (Figs. 3 and 7), the crossing-time delta N of Sec. V
//! (Fig. 11), the spectra of Sec. IV (G3), and the tree-level f_NL split at
//! a slice (G4). The tests are expensive in debug builds; run them with
//! `cargo test --release -- --ignored`. See
//! `docs/general-nonminimal-inflation.md`.

use phys_type::{
    BackgroundSurvey, ConformalPerturbationTolerance, ErrorTolerances, Formulation, Mass,
    MomentumTriangle, SpectrumRunSettings, SplitSettings, curvature_correlators,
};
use phys_type_models::kms::{
    KmsSystem, PHI_START, TRAJECTORIES, at_rest, benchmark_system, crossing_delta_n,
    slow_roll_tilt, transfer_functions, turn_peak,
};

fn settings() -> SpectrumRunSettings {
    SpectrumRunSettings {
        pivot_e_folds_before_end: 60.0,
        e_folds_before_exit: 6.0,
        background_sample_every: 1,
        ..SpectrumRunSettings::default()
    }
}

fn survey(system: &KmsSystem, chi0: f64) -> BackgroundSurvey<2> {
    system
        .survey(at_rest(PHI_START, chi0), &settings())
        .unwrap()
}

#[test]
#[ignore = "expensive; run in release mode"]
fn the_trajectories_fall_off_the_ridge_as_in_kms_figs_3_and_7() {
    let system = benchmark_system(Formulation::Metric);
    let surveys = TRAJECTORIES.map(|chi0| survey(&system, chi0));
    // Fig. 3: N_total = 71.5 for trajectory 2.
    let total = surveys[1].e_folds_at_end;
    assert!((total - 71.5).abs() < 0.05, "trajectory 2: {total} e-folds");
    // Fig. 7: the turn rate peaks 34.5 and 4.9 e-folds before the end; the
    // exact evolution gives 34.06 and 5.15, a spacing of 28.9 against 29.6.
    let (peak_1, before_1) = turn_peak(&surveys[0]).unwrap();
    let (peak_2, before_2) = turn_peak(&surveys[1]).unwrap();
    assert!(peak_1 > 0.5 && peak_2 > 0.5, "{peak_1}, {peak_2}");
    assert!(
        (before_1 - 34.5).abs() < 1.0,
        "trajectory 1 peak at {before_1}"
    );
    assert!(
        (before_2 - 4.9).abs() < 0.5,
        "trajectory 2 peak at {before_2}"
    );
    // Trajectory 3 stays on the ridge until inflation ends.
    assert_eq!(turn_peak(&surveys[2]), None);
    // Trajectory 2 nearly stops inflating at the turn: epsilon_H reaches
    // 0.99 there, which is why its f_NL is large and steep in chi(tau_0).
    let turning = surveys[1]
        .samples
        .iter()
        .filter(|sample| surveys[1].e_folds_at_end - sample.e_folds > 1.0)
        .map(|sample| sample.epsilon_h)
        .fold(0.0, f64::max);
    assert!(
        turning > 0.98 && turning < 1.0,
        "epsilon_H at the turn {turning}"
    );
}

#[test]
#[ignore = "expensive; run in release mode"]
fn the_crossing_delta_n_reproduces_kms_fig_11() {
    let system = benchmark_system(Formulation::Metric);
    for (index, chi0) in TRAJECTORIES.into_iter().enumerate() {
        let survey = survey(&system, chi0);
        let delta_n = crossing_delta_n(&system, &survey, &settings(), 60.0).unwrap();
        if index == 1 {
            // Sec. V: |f_NL| = 43.3 for trajectory 2.
            assert!(
                (delta_n.f_nl.abs() - 43.3).abs() < 0.3,
                "trajectory 2: f_NL {}",
                delta_n.f_nl
            );
        } else {
            assert!(
                delta_n.f_nl.abs() < 0.1,
                "trajectory {}: f_NL {}",
                index + 1,
                delta_n.f_nl
            );
        }
        // The linear part against the exact mode solver at the end: the
        // slow-roll velocities at crossing cost about 5%.
        let pivot = survey.trajectory.state_at(survey.e_folds_at_end - 60.0);
        let k = (pivot.a * system.hubble(&pivot)).into_value();
        let exact = system
            .mode_spectrum(
                &settings(),
                &survey.trajectory,
                Mass::new(k),
                survey.e_folds_at_end - 0.01,
            )
            .unwrap()
            .spectra
            .curvature
            .into_value();
        assert!(
            (delta_n.power / exact - 1.0).abs() < 0.06,
            "trajectory {}: delta N {} vs mode solver {exact}",
            index + 1,
            delta_n.power
        );
    }
}

#[test]
#[ignore = "expensive; run in release mode"]
fn the_split_f_nl_converges_for_all_three_trajectories() {
    // G4. The transport carried to the end of inflation does not converge
    // on trajectories 1 and 2: the correlators grow during the turn and then
    // cancel. Split at a slice before that growth, the result is the same on
    // every slice. The background must be solved to 1e-13: at the default
    // 1e-10, f_NL of trajectory 2 is 1.5% off (appendix F of the note).
    let system = benchmark_system(Formulation::Metric);
    let precise = SpectrumRunSettings {
        tolerances: ErrorTolerances::new(
            ConformalPerturbationTolerance::<2>::background_in_planck_units(1.0e-23),
            1.0e-13,
        ),
        background_sample_every: usize::MAX,
        ..settings()
    };
    let split = SplitSettings::default();
    // (slices in e-folds before the end, expected f_NL): the slices precede
    // the growth, which starts near 34 e-folds before the end on trajectory 1
    // and near 10 on trajectory 2.
    let cases = [
        ([50.0, 45.0], 0.013_545),
        ([20.0, 10.0], 38.287),
        ([20.0, 10.0], 0.013_530),
    ];
    for (index, (chi0, (before, expected))) in TRAJECTORIES.into_iter().zip(cases).enumerate() {
        let survey = system.survey(at_rest(PHI_START, chi0), &precise).unwrap();
        let end = survey.e_folds_at_end;
        let pivot = survey.trajectory.state_at(end - 60.0);
        let k = (pivot.a * system.hubble(&pivot)).into_value();
        let triangle = MomentumTriangle::try_new(k, k, k).unwrap();
        let output = end - 0.01;
        let f_nl: Vec<f64> = system
            .split_curvature_correlators(
                &survey.trajectory,
                triangle,
                &before.map(|n| end - n),
                output,
                &split,
            )
            .unwrap()
            .iter()
            .map(|correlators| correlators.reduced_f_nl())
            .collect();
        let label = index + 1;
        assert!(
            (f_nl[0] - f_nl[1]).abs() < 1.0e-6 * f_nl[0].abs(),
            "trajectory {label}: {f_nl:?} on two slices"
        );
        assert!(
            (f_nl[0] - expected).abs() < 1.0e-3 * expected,
            "trajectory {label}: f_NL {} against {expected}",
            f_nl[0]
        );
    }
}

#[test]
#[ignore = "expensive; run in release mode"]
fn the_split_matches_the_transport_where_the_transport_converges() {
    // On trajectory 2, the transport converges up to 3 e-folds before the
    // end; there the split must reproduce it.
    let system = benchmark_system(Formulation::Metric);
    let survey = survey(&system, TRAJECTORIES[1]);
    let end = survey.e_folds_at_end;
    let pivot = survey.trajectory.state_at(end - 60.0);
    let k = (pivot.a * system.hubble(&pivot)).into_value();
    let triangle = MomentumTriangle::try_new(k, k, k).unwrap();
    let output = end - 3.0;
    let split = SplitSettings::default();
    let transport = system
        .three_point_state(
            &survey.trajectory,
            triangle,
            split.e_folds_before_exit,
            output,
            split.relative_tolerance,
            split.max_steps,
        )
        .unwrap();
    let direct = curvature_correlators(
        &system,
        &survey.trajectory.state_at(output),
        triangle,
        &transport,
        split.maximum_output_k_over_ah,
    )
    .unwrap()
    .reduced_f_nl();
    let split = system
        .split_curvature_correlators(&survey.trajectory, triangle, &[end - 10.0], output, &split)
        .unwrap()[0]
        .reduced_f_nl();
    // 40.9458 in both, issue #67.
    assert!(
        (split - direct).abs() < 1.0e-5 * direct.abs(),
        "split {split} vs transport {direct}"
    );
    assert!((direct - 40.9458).abs() < 1.0e-3, "transport {direct}");
}

#[test]
#[ignore = "expensive; run in release mode"]
fn the_spectra_at_n_60_reproduce_kms_sec_iv() {
    // G3. KMS Sec. IV: n_s(t_*) = 0.967 for all three trajectories from the
    // slow-roll Eq. (85), and T_RS from N_* = 60 to 40 e-folds before the
    // end of 0.530, 0.011, and 0.001, from Eqs. (70), (72), and (77).
    let system = benchmark_system(Formulation::Metric);
    let precise = SpectrumRunSettings {
        tolerances: ErrorTolerances::new(
            ConformalPerturbationTolerance::<2>::background_in_planck_units(1.0e-23),
            1.0e-13,
        ),
        background_sample_every: usize::MAX,
        ..settings()
    };
    let transfer = [0.530, 0.011, 0.001];
    for (index, (chi0, expected)) in TRAJECTORIES.into_iter().zip(transfer).enumerate() {
        let label = index + 1;
        let survey = system.survey(at_rest(PHI_START, chi0), &precise).unwrap();
        let tilt = slow_roll_tilt(&system, &survey, 60.0).unwrap();
        assert!(
            (tilt - 0.967).abs() < 5.0e-4,
            "trajectory {label}: n_s(t_*) {tilt}"
        );
        let (t_rs, _) = transfer_functions(&system, &survey, 60.0, 40.0, 1.0e-3).unwrap();
        assert!(
            (t_rs - expected).abs() < 5.0e-4,
            "trajectory {label}: T_RS {t_rs} against {expected}"
        );
        // The exact spectrum 5 e-folds after crossing has the same tilt as
        // Eq. (85) to the next slow-roll order (0.9677--0.9678).
        let tilt_at = |before: f64| {
            let end = survey.e_folds_at_end;
            let pivot = survey.trajectory.state_at(end - 60.0);
            let k = (pivot.a * system.hubble(&pivot)).into_value();
            let curvature = |kk: f64| {
                system
                    .mode_spectrum(&precise, &survey.trajectory, Mass::new(kk), end - before)
                    .unwrap()
                    .spectra
                    .curvature
                    .into_value()
            };
            let step: f64 = 0.02;
            1.0 + (curvature(k * step.exp()) / curvature(k * (-step).exp())).ln() / (2.0 * step)
        };
        let exact = tilt_at(55.0);
        assert!(
            (exact - tilt).abs() < 1.0e-3,
            "trajectory {label}: n_s {exact} 55 e-folds before the end against {tilt}"
        );
        // On trajectory 2 the turn 5 e-folds before the end tilts the
        // spectrum read off at the end of inflation to 0.936; KMS Fig. 10
        // shows n_s(t_*) only.
        if index == 1 {
            let at_end = tilt_at(0.01);
            assert!(
                (at_end - 0.936_16).abs() < 1.0e-4,
                "n_s at the end {at_end}"
            );
        }
    }
}
