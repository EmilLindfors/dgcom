//! 3D Coriolis force.
//!
//! Applies Coriolis force to 3D velocity field.
//!
//! du/dt = f * v
//! dv/dt = -f * u
//!
//! where f is the Coriolis parameter.

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::core::blocks::{Blocks, for_each_block};
use crate::solver::state::Solution3D;
use crate::source::CoriolisSource2D;
use crate::types::ElementIndex;

/// Apply Coriolis force to 3D velocity field.
///
/// Adds the Coriolis tendency to the explicit RHS.
///
/// # Arguments
/// * `rhs` - Right-hand side state (accumulates tendency).
/// * `state` - Current state (u, v).
/// * `mesh` - 2D mesh (for y-coordinates).
/// * `ops` - 2D operators (for nodes).
/// * `coriolis` - Coriolis parameter configuration.
pub fn apply_coriolis_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    coriolis: &CoriolisSource2D,
    elements: Blocks,
) {
    let (n_nodes, n_levels) = (ops.n_nodes, state.n_levels);
    let n = mesh.n_elements * n_nodes * n_levels;
    for_each_block(
        elements,
        [&mut rhs.u[..n], &mut rhs.v[..n]],
        || (),
        |_, k, [rhs_u, rhs_v]| {
            let k = ElementIndex::new(k);
            for i in 0..n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [_x, y] = mesh.reference_to_physical(k, r, s);
                let f = coriolis.f_at(y);
                let (u_col, v_col) = (state.u_column(k, i), state.v_column(k, i));
                let column = i * n_levels..(i + 1) * n_levels;
                // du/dt = f v, dv/dt = −f u
                for ((du, dv), (&u, &v)) in rhs_u[column.clone()]
                    .iter_mut()
                    .zip(&mut rhs_v[column])
                    .zip(u_col.iter().zip(v_col))
                {
                    *du += f * v;
                    *dv -= f * u;
                }
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::operators::GeometricFactors2D;

    #[test]
    fn test_coriolis_3d_f_plane() {
        let n_elem = 1;
        let n_nodes = 1;
        let n_levels = 2;

        // Mock mesh/ops
        // We need a real mesh for physical coordinates
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(0); // 1 node (order 0)

        let mut state = Solution3D::new(n_elem, n_nodes, n_levels);
        let mut rhs = Solution3D::new(n_elem, n_nodes, n_levels);

        // Set u=10, v=0
        state.u.fill(10.0);
        state.v.fill(0.0);

        let f0 = 1.0e-4;
        let coriolis = CoriolisSource2D::f_plane(f0);

        apply_coriolis_3d(
            &mut rhs,
            &state,
            &mesh,
            &ops,
            &coriolis,
            mesh.n_elements.into(),
        );

        // du/dt = f*v = 0
        // dv/dt = -f*u = -1e-4 * 10 = -1e-3

        assert!((rhs.u[0] - 0.0).abs() < 1e-10);
        assert!((rhs.v[0] - (-1.0e-3)).abs() < 1e-10);
        assert!((rhs.v[1] - (-1.0e-3)).abs() < 1e-10);
    }
}
