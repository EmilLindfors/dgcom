//! Polynomial evaluation and node generation.
//!
//! This module provides:
//! - 1D Legendre polynomials and their derivatives
//! - 1D Gauss-Lobatto-Legendre (GLL) nodes and weights
//! - 2D tensor-product Legendre polynomials for quadrilateral elements
//! - 2D tensor-product GLL nodes and weights

mod lagrange;
mod legendre;
mod legendre_2d;
mod nodes;

pub use lagrange::{lagrange_basis, lagrange_basis_derivative};
pub use legendre::{legendre, legendre_and_derivative, legendre_derivative};
pub use legendre_2d::{
    legendre_2d, legendre_2d_gradient, legendre_2d_gradient_normalized, legendre_2d_norm,
    legendre_2d_normalized, legendre_2d_normalized_with_gradient, legendre_2d_with_gradient,
    mode_degrees, mode_index, node_index_1d_to_2d, node_index_2d_to_1d, tensor_product_gll_nodes,
    tensor_product_gll_weights,
};
pub use nodes::{gauss_legendre_nodes_weights, gauss_lobatto_nodes, gauss_lobatto_weights};
