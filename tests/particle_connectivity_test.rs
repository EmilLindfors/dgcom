//! Gates for connectivity statistics from tracked particles
//! (`dgcom::particles::connectivity`, TODO F.2).
//!
//! - A uniform current carries particles from one cage through another: each
//!   one's exposure is its chord through the receiving footprint over the
//!   speed, and its first contact the distance to the footprint, to the step.
//! - A random walk from a point: the mean occupation time of a disc around it
//!   is the walk's exact `Σ Δt (1 − e^{−a²/4Kt_n})`.
//! - Independent particles: the share's spread across seeds is the binomial
//!   error, two ensembles of the same layout agree within their standard
//!   errors, and a layout with the receiver further away differs by many.

use dgcom::mesh::Mesh2D;
use dgcom::operators::DGOperators2D;
use dgcom::particles::{
    ConnectivityEnsemble, ConnectivityMatrix, ConnectivityRecorder, ContactZone, NodalVelocity2D,
    Particle2D, ParticleTracker2D,
};

/// Track `particles` for `n` steps of `dt` through `field`, recording each
/// at the surface after every step (particle `i` is the recorder's `i`).
fn track(
    tracker: &ParticleTracker2D,
    field: &NodalVelocity2D,
    particles: &mut [Particle2D],
    recorder: &mut ConnectivityRecorder,
    n: usize,
    dt: f64,
) {
    for step in 0..n {
        let t = step as f64 * dt;
        tracker.step(particles, field, t, dt);
        for (i, p) in particles.iter().enumerate() {
            recorder.record(i, t + dt, dt, p.position(), || 0.0, 1.0);
        }
    }
}

#[test]
fn passage_exposure_is_the_chord_time() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 500.0, 0.0, 500.0, 10, 10);
    let ops = DGOperators2D::new(2);
    let speed = 0.5;
    let u = vec![speed; mesh.n_elements * ops.n_nodes];
    let v = vec![0.0; u.len()];
    let field = NodalVelocity2D::new(&u, &v, ops.n_nodes);
    let (radius, source, receiver) = (25.0, [100.0, 250.0], [300.0, 250.0]);
    for dt in [4.0, 1.0] {
        let tracker = ParticleTracker2D::new(&mesh, &ops);
        let mut recorder = ConnectivityRecorder::new(vec![
            ContactZone::circular(source, radius, [0.0, 5.0]),
            ContactZone::circular(receiver, radius, [0.0, 5.0]),
        ]);
        // A sunflower over the source's footprint
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut particles: Vec<Particle2D> = (0..200)
            .map(|i| {
                let r = 0.99 * radius * ((i as f64 + 0.5) / 200.0).sqrt();
                let a = i as f64 * golden;
                let p = [source[0] + r * a.cos(), source[1] + r * a.sin()];
                recorder.add(0, 0, 0.0);
                tracker.release(i, p).unwrap()
            })
            .collect();
        let start: Vec<[f64; 2]> = particles.iter().map(Particle2D::position).collect();
        track(
            &tracker,
            &field,
            &mut particles,
            &mut recorder,
            (520.0 / dt) as usize,
            dt,
        );
        let mut worst = [0.0_f64; 2];
        for (i, [x, y]) in start.into_iter().enumerate() {
            let half_chord = (radius * radius - (y - receiver[1]).powi(2)).sqrt();
            let contacts = recorder.particle_contacts(i);
            assert_eq!(contacts.len(), 1, "particle {i}: {contacts:?}");
            let c = contacts[0];
            assert_eq!(c.zone, 1);
            worst[0] = worst[0].max((c.exposure - 2.0 * half_chord / speed).abs());
            // First contact: the end of the step that crosses into it
            let arrival = (receiver[0] - half_chord - x) / speed;
            worst[1] = worst[1].max(c.first - arrival);
            assert!(c.first >= arrival - 1e-9);
        }
        println!(
            "dt {dt} s: exposure within {:.3} s of the chord time, first contact within {:.3} s",
            worst[0], worst[1]
        );
        assert!(worst[0] <= dt + 1e-9 && worst[1] <= dt + 1e-9);
        let m = recorder.matrix(0.0, |_| true);
        assert_eq!((m.share(0, 1), m.share(0, 0)), (1.0, 0.0));
    }
}

#[test]
fn random_walk_occupation_time_is_exact() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 1000.0, 8, 8);
    let ops = DGOperators2D::new(1);
    let zero = vec![0.0; mesh.n_elements * ops.n_nodes];
    let field = NodalVelocity2D::new(&zero, &zero, ops.n_nodes);
    let (k, dt, n_steps, a) = (1.0, 1.0, 400, 20.0);
    let center = [500.0, 500.0];
    let tracker = ParticleTracker2D::new(&mesh, &ops)
        .with_diffusivity(k)
        .with_seed(7);
    // A point source, and a disc of radius `a` around it as the receiver
    let mut recorder = ConnectivityRecorder::new(vec![
        ContactZone::circular(center, 1e-6, [0.0, 1.0]),
        ContactZone::circular(center, a, [0.0, 1.0]),
    ]);
    let n = 4000;
    let mut particles: Vec<Particle2D> = (0..n)
        .map(|i| {
            recorder.add(0, 0, 0.0);
            tracker.release(i, center).unwrap()
        })
        .collect();
    track(&tracker, &field, &mut particles, &mut recorder, n_steps, dt);
    // After n steps the walk is exactly N(0, 2K nΔt) per component
    let exact: f64 = (1..=n_steps)
        .map(|s| dt * (1.0 - (-a * a / (4.0 * k * s as f64 * dt)).exp()))
        .sum();
    let exposures: Vec<f64> = (0..n as usize)
        .map(|i| {
            recorder
                .particle_contacts(i)
                .iter()
                .find(|c| c.zone == 1)
                .map_or(0.0, |c| c.exposure)
        })
        .collect();
    let mean = exposures.iter().sum::<f64>() / n as f64;
    let sd = (exposures.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let se = sd / (n as f64).sqrt();
    println!(
        "occupation time of the disc: {mean:.2} ± {se:.2} s, exact {exact:.2} s ({:+.1} SE)",
        (mean - exact) / se
    );
    assert!((mean - exact).abs() < 4.0 * se);
    let m = recorder.matrix(0.0, |_| true);
    assert!((m.mean_exposure(0, 1) - mean).abs() < 1e-9);
}

/// The matrices of `seeds` runs of `n` particles walking from a point, with
/// a receiver of radius 10 m `distance` away.
fn walk_ensemble(distance: f64, seeds: std::ops::Range<u64>, n: u64) -> ConnectivityEnsemble {
    let mesh = Mesh2D::uniform_rectangle(0.0, 600.0, 0.0, 600.0, 6, 6);
    let ops = DGOperators2D::new(1);
    let zero = vec![0.0; mesh.n_elements * ops.n_nodes];
    let field = NodalVelocity2D::new(&zero, &zero, ops.n_nodes);
    let center = [300.0, 300.0];
    let members: Vec<ConnectivityMatrix> = seeds
        .map(|seed| {
            let tracker = ParticleTracker2D::new(&mesh, &ops)
                .with_diffusivity(1.0)
                .with_seed(seed);
            let mut recorder = ConnectivityRecorder::new(vec![
                ContactZone::circular(center, 5.0, [0.0, 1.0]),
                ContactZone::circular([center[0] + distance, center[1]], 10.0, [0.0, 1.0]),
            ]);
            let mut particles: Vec<Particle2D> = (0..n)
                .map(|i| {
                    recorder.add(0, 0, 0.0);
                    tracker.release(i, center).unwrap()
                })
                .collect();
            track(&tracker, &field, &mut particles, &mut recorder, 300, 2.0);
            recorder.matrix(0.0, |_| true)
        })
        .collect();
    ConnectivityEnsemble::new(members)
}

#[test]
fn seed_spread_tells_layouts_apart() {
    let n = 200;
    let near = walk_ensemble(40.0, 0..40, n);
    let binomial = near
        .members()
        .iter()
        .map(|m| m.binomial_error(0, 1))
        .sum::<f64>()
        / near.members().len() as f64;
    println!(
        "share {:.3}, spread across 40 seeds {:.4}, binomial {:.4}",
        near.mean(0, 1),
        near.spread(0, 1),
        binomial
    );
    assert!(near.mean(0, 1) > 0.1 && near.mean(0, 1) < 0.9);
    // Independent particles: the spread is the binomial error (the sample
    // standard deviation of 40 is good to ≈ 11 %)
    assert!((near.spread(0, 1) / binomial - 1.0).abs() < 0.35);
    // The same layout with other seeds: no difference beyond the noise
    let again = walk_ensemble(40.0, 100..140, n);
    let same = near.difference(&again, 0, 1);
    // The receiver further away: fewer arrive, far beyond the noise
    let far = walk_ensemble(55.0, 200..240, n);
    let other = near.difference(&far, 0, 1);
    println!(
        "same layout: {:+.4} ± {:.4} (z {:+.1}); receiver 15 m further: {:+.4} ± {:.4} (z {:+.1})",
        same.estimate,
        same.standard_error,
        same.z(),
        other.estimate,
        other.standard_error,
        other.z()
    );
    assert!(same.z().abs() < 4.0);
    assert!(other.z() > 8.0);
}
