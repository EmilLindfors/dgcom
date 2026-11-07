//! Strongly-typed index newtypes.
//!
//! These types prevent mixing up different kinds of indices
//! (element vs face vs node vs level).

use std::fmt;

/// Macro to generate index newtypes with common functionality.
macro_rules! define_index {
    (
        $(#[$meta:meta])*
        $name:ident, $display_prefix:literal
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(usize);

        impl $name {
            /// Create a new index.
            #[inline]
            pub const fn new(index: usize) -> Self {
                Self(index)
            }

            /// Get the raw index value.
            #[inline]
            pub const fn get(self) -> usize {
                self.0
            }

            /// Convert to usize.
            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0
            }

            /// First index (0).
            pub const ZERO: Self = Self(0);

            /// Increment index by one.
            #[inline]
            pub fn next(self) -> Self {
                Self(self.0 + 1)
            }

            /// Decrement index by one, saturating at zero.
            #[inline]
            pub fn prev(self) -> Self {
                Self(self.0.saturating_sub(1))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $display_prefix, self.0)
            }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(index: usize) -> Self {
                Self(index)
            }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(idx: $name) -> usize {
                idx.0
            }
        }

        // Allow using as array index
        impl<T> std::ops::Index<$name> for [T] {
            type Output = T;
            #[inline]
            fn index(&self, idx: $name) -> &T {
                &self[idx.0]
            }
        }

        impl<T> std::ops::IndexMut<$name> for [T] {
            #[inline]
            fn index_mut(&mut self, idx: $name) -> &mut T {
                &mut self[idx.0]
            }
        }

        impl<T> std::ops::Index<$name> for Vec<T> {
            type Output = T;
            #[inline]
            fn index(&self, idx: $name) -> &T {
                &self[idx.0]
            }
        }

        impl<T> std::ops::IndexMut<$name> for Vec<T> {
            #[inline]
            fn index_mut(&mut self, idx: $name) -> &mut T {
                &mut self[idx.0]
            }
        }
    };
}

define_index!(
    /// Element index in a mesh.
    ///
    /// Used to identify elements (cells) in 1D and 2D meshes.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::ElementIndex;
    ///
    /// let elem = ElementIndex::new(42);
    /// assert_eq!(elem.get(), 42);
    /// ```
    ElementIndex,
    "E"
);

define_index!(
    /// Face/edge index in a mesh.
    ///
    /// Used to identify faces (1D) or edges (2D) in a mesh.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::FaceIndex;
    ///
    /// let face = FaceIndex::new(10);
    /// assert_eq!(face.get(), 10);
    /// ```
    FaceIndex,
    "F"
);

define_index!(
    /// Node index within an element.
    ///
    /// Used to identify quadrature/interpolation nodes within an element.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::NodeIndex;
    ///
    /// let node = NodeIndex::new(3);
    /// assert_eq!(node.get(), 3);
    /// ```
    NodeIndex,
    "N"
);

define_index!(
    /// Vertical level index in a sigma grid.
    ///
    /// Used to identify vertical layers in 3D simulations.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::LevelIndex;
    ///
    /// let level = LevelIndex::new(5);
    /// assert_eq!(level.get(), 5);
    /// ```
    LevelIndex,
    "L"
);

// =============================================================================
// Iterator support
// =============================================================================

impl ElementIndex {
    /// Create an iterator over [0, n) element indices.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::ElementIndex;
    ///
    /// let indices: Vec<_> = ElementIndex::iter(5).collect();
    /// assert_eq!(indices.len(), 5);
    /// assert_eq!(indices[0].get(), 0);
    /// assert_eq!(indices[4].get(), 4);
    /// ```
    pub fn iter(n: usize) -> impl ExactSizeIterator<Item = ElementIndex> {
        (0..n).map(ElementIndex)
    }

    /// Create an iterator over [start, end) element indices.
    pub fn range_iter(
        start: ElementIndex,
        end: ElementIndex,
    ) -> impl ExactSizeIterator<Item = ElementIndex> {
        (start.0..end.0).map(ElementIndex)
    }
}

impl NodeIndex {
    /// Create an iterator over [0, n) node indices.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::NodeIndex;
    ///
    /// let indices: Vec<_> = NodeIndex::iter(9).collect();
    /// assert_eq!(indices.len(), 9);
    /// assert_eq!(indices[0].get(), 0);
    /// assert_eq!(indices[8].get(), 8);
    /// ```
    pub fn iter(n: usize) -> impl ExactSizeIterator<Item = NodeIndex> {
        (0..n).map(NodeIndex)
    }
}

impl FaceIndex {
    /// Create an iterator over [0, n) face indices.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::FaceIndex;
    ///
    /// let indices: Vec<_> = FaceIndex::iter(3).collect();
    /// assert_eq!(indices.len(), 3);
    /// assert_eq!(indices[0].get(), 0);
    /// assert_eq!(indices[2].get(), 2);
    /// ```
    pub fn iter(n: usize) -> impl ExactSizeIterator<Item = FaceIndex> {
        (0..n).map(FaceIndex)
    }
}

impl LevelIndex {
    /// Create an iterator over [0, n) level indices.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::types::LevelIndex;
    ///
    /// let indices: Vec<_> = LevelIndex::iter(10).collect();
    /// assert_eq!(indices.len(), 10);
    /// assert_eq!(indices[0].get(), 0);
    /// assert_eq!(indices[9].get(), 9);
    /// ```
    pub fn iter(n: usize) -> impl ExactSizeIterator<Item = LevelIndex> {
        (0..n).map(LevelIndex)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_index() {
        let idx = ElementIndex::new(42);
        assert_eq!(idx.get(), 42);
        assert_eq!(idx.as_usize(), 42);
        assert_eq!(usize::from(idx), 42);
    }

    #[test]
    fn test_index_arithmetic() {
        let idx = ElementIndex::new(5);
        assert_eq!(idx.next().get(), 6);
        assert_eq!(idx.prev().get(), 4);

        // Saturating at zero
        assert_eq!(ElementIndex::ZERO.prev().get(), 0);
    }

    #[test]
    fn test_array_indexing() {
        let data = [10, 20, 30, 40, 50];
        let idx = ElementIndex::new(2);
        assert_eq!(data[idx], 30);
    }

    #[test]
    fn test_array_indexing_mut() {
        let mut data = [10, 20, 30, 40, 50];
        let idx = ElementIndex::new(2);
        data[idx] = 100;
        assert_eq!(data[2], 100);
    }

    #[test]
    fn test_element_index_iter() {
        let indices: Vec<_> = ElementIndex::iter(5).collect();
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0].get(), 0);
        assert_eq!(indices[4].get(), 4);
    }

    #[test]
    fn test_element_index_range_iter() {
        let start = ElementIndex::new(3);
        let end = ElementIndex::new(7);
        let indices: Vec<_> = ElementIndex::range_iter(start, end).collect();
        assert_eq!(indices.len(), 4);
        assert_eq!(indices[0].get(), 3);
        assert_eq!(indices[3].get(), 6);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ElementIndex::new(42)), "E42");
        assert_eq!(format!("{}", FaceIndex::new(10)), "F10");
        assert_eq!(format!("{}", NodeIndex::new(3)), "N3");
        assert_eq!(format!("{}", LevelIndex::new(5)), "L5");
    }

    #[test]
    fn test_from_conversions() {
        let elem: ElementIndex = 42.into();
        assert_eq!(elem.get(), 42);

        let back: usize = elem.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn test_node_index_iter() {
        let indices: Vec<_> = NodeIndex::iter(9).collect();
        assert_eq!(indices.len(), 9);
        assert_eq!(indices[0].get(), 0);
        assert_eq!(indices[8].get(), 8);
    }

    #[test]
    fn test_face_index_iter() {
        let indices: Vec<_> = FaceIndex::iter(3).collect();
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0].get(), 0);
        assert_eq!(indices[2].get(), 2);
    }

    #[test]
    fn test_level_index_iter() {
        let indices: Vec<_> = LevelIndex::iter(10).collect();
        assert_eq!(indices.len(), 10);
        assert_eq!(indices[0].get(), 0);
        assert_eq!(indices[9].get(), 9);
    }
}
