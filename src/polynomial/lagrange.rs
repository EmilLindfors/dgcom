//! Lagrange interpolation through a node set.
//!
//! The nodal DG basis is the Lagrange basis on the element nodes
//! (Hesthaven & Warburton 2008, §3.1):
//!
//! ```text
//! ℓᵢ(x) = Πⱼ≠ᵢ (x − xⱼ) / (xᵢ − xⱼ),        u_h(x) = Σᵢ uᵢ ℓᵢ(x)
//! ```
//!
//! so a nodal field is evaluated anywhere in the element by weighting its
//! nodal values with the basis values at the point. The product form costs
//! O(n²) per point and is exact at the nodes (`ℓᵢ(xⱼ) = δᵢⱼ` bit for bit);
//! for the handful of nodes of a DG element that beats the barycentric form.
//! The derivative `ℓᵢ'(x) = Σₘ≠ᵢ 1/(xᵢ − xₘ) Πⱼ≠ᵢ,ₘ (x − xⱼ)/(xᵢ − xⱼ)` (the
//! product rule) costs O(n³), and differentiates `u_h` exactly.

/// Values `ℓᵢ(x)` of the Lagrange basis on `nodes` (distinct) at `x`, into
/// `out` (same length as `nodes`).
pub fn lagrange_basis(nodes: &[f64], x: f64, out: &mut [f64]) {
    assert_eq!(nodes.len(), out.len(), "one basis value per node");
    for (i, (&xi, l)) in nodes.iter().zip(out.iter_mut()).enumerate() {
        *l = nodes
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &xj)| (x - xj) / (xi - xj))
            .product();
    }
}

/// Derivatives `ℓᵢ'(x)` of the Lagrange basis on `nodes` (distinct) at `x`,
/// into `out` (same length as `nodes`).
pub fn lagrange_basis_derivative(nodes: &[f64], x: f64, out: &mut [f64]) {
    assert_eq!(nodes.len(), out.len(), "one basis derivative per node");
    for (i, (&xi, d)) in nodes.iter().zip(out.iter_mut()).enumerate() {
        *d = nodes
            .iter()
            .enumerate()
            .filter(|&(m, _)| m != i)
            .map(|(m, &xm)| {
                let rest: f64 = nodes
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i && j != m)
                    .map(|(_, &xj)| (x - xj) / (xi - xj))
                    .product();
                rest / (xi - xm)
            })
            .sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::gauss_lobatto_nodes;

    /// The basis is a partition of unity, the identity at the nodes, and
    /// interpolates polynomials up to degree N exactly.
    #[test]
    fn lagrange_basis_is_exact_for_degree_n() {
        for order in 1..=6 {
            let nodes = gauss_lobatto_nodes(order);
            let mut l = vec![0.0; nodes.len()];
            for (i, &xi) in nodes.iter().enumerate() {
                lagrange_basis(&nodes, xi, &mut l);
                for (j, &lj) in l.iter().enumerate() {
                    assert_eq!(lj, if i == j { 1.0 } else { 0.0 });
                }
            }
            let p = |x: f64| {
                (0..=order)
                    .map(|k| (k as f64 + 0.5) * x.powi(k as i32))
                    .sum::<f64>()
            };
            let values: Vec<f64> = nodes.iter().map(|&x| p(x)).collect();
            for x in [-1.0, -0.73, -0.1, 0.0, 0.42, 0.999, 1.0] {
                lagrange_basis(&nodes, x, &mut l);
                assert!((l.iter().sum::<f64>() - 1.0).abs() < 1e-13);
                let interpolated: f64 = l.iter().zip(&values).map(|(a, b)| a * b).sum();
                assert!(
                    (interpolated - p(x)).abs() < 1e-12,
                    "order {order}, x {x}: {interpolated} vs {}",
                    p(x)
                );
            }
        }
    }

    /// The derivatives sum to 0 and differentiate polynomials up to degree N
    /// exactly.
    #[test]
    fn lagrange_basis_derivative_is_exact_for_degree_n() {
        for order in 1..=6 {
            let nodes = gauss_lobatto_nodes(order);
            let mut d = vec![0.0; nodes.len()];
            let p = |x: f64| {
                (0..=order)
                    .map(|k| (k as f64 + 0.5) * x.powi(k as i32))
                    .sum::<f64>()
            };
            let dp = |x: f64| {
                (1..=order)
                    .map(|k| (k as f64 + 0.5) * k as f64 * x.powi(k as i32 - 1))
                    .sum::<f64>()
            };
            let values: Vec<f64> = nodes.iter().map(|&x| p(x)).collect();
            for x in [-1.0, -0.73, -0.1, 0.0, 0.42, 0.999, 1.0] {
                lagrange_basis_derivative(&nodes, x, &mut d);
                assert!(d.iter().sum::<f64>().abs() < 1e-11);
                let derivative: f64 = d.iter().zip(&values).map(|(a, b)| a * b).sum();
                assert!(
                    (derivative - dp(x)).abs() < 1e-10,
                    "order {order}, x {x}: {derivative} vs {}",
                    dp(x)
                );
            }
        }
    }
}
