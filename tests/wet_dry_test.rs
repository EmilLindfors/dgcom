use dgcom::{
    DGOperators2D, Depth, ElementIndex, GeometricFactors2D, Mesh2D, SWESolution2D, SWEState2D,
    WetDryConfig, apply_wet_dry_correction_all, swe_positivity_limiter_2d,
};

/// Geometry of one unit-square element.
fn unit_geometry(ops: &DGOperators2D) -> GeometricFactors2D {
    GeometricFactors2D::compute(&Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1), ops)
}

fn weighted_mass(solution: &SWESolution2D, ops: &DGOperators2D) -> f64 {
    let k = ElementIndex::new(0);
    solution
        .element_h(k)
        .iter()
        .zip(ops.weights.iter())
        .map(|(&h, &w)| h * w)
        .sum()
}

#[test]
fn test_positivity_dry_average_does_not_inject_mass() {
    let ops = DGOperators2D::new(2);
    let mut swe = SWESolution2D::new(1, ops.n_nodes);
    let k = ElementIndex::new(0);

    for i in 0..ops.n_nodes {
        swe.set_state(k, i, SWEState2D::new(0.005, 0.1, -0.1));
    }

    let initial_mass = weighted_mass(&swe, &ops);
    swe_positivity_limiter_2d(&mut swe, &unit_geometry(&ops), 0.01);
    let final_mass = weighted_mass(&swe, &ops);

    assert!((final_mass - initial_mass).abs() < 1e-14);
    for i in 0..ops.n_nodes {
        let state = swe.get_state(k, i);
        assert!((state.h - 0.005).abs() < 1e-14);
        assert!(state.hu.abs() < 1e-14);
        assert!(state.hv.abs() < 1e-14);
    }
}

#[test]
fn test_wet_dry_correction_preserves_nonnegative_element_mass() {
    let ops = DGOperators2D::new(2);
    let config = WetDryConfig::new(Depth::new(0.01), 9.81);
    let mut solution = SWESolution2D::new(1, ops.n_nodes);
    let k = ElementIndex::new(0);

    for i in 0..ops.n_nodes {
        let h = if i == 0 { -0.01 } else { 0.02 };
        solution.set_state(k, i, SWEState2D::new(h, 0.2, -0.1));
    }

    let initial_mass = weighted_mass(&solution, &ops).max(0.0);
    apply_wet_dry_correction_all(&mut solution, &unit_geometry(&ops), &config);
    let final_mass = weighted_mass(&solution, &ops);

    assert!(solution.element_h(k).iter().all(|&h| h >= 0.0));
    assert!((final_mass - initial_mass).abs() < 1e-12);
}
