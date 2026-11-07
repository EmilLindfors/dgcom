//! Side boundary types with named fields.
//!
//! Provides strongly-typed structures for per-side boundary specifications,
//! eliminating the need to remember array index conventions.

use std::fmt;

/// Boundary specification with named fields for each side.
///
/// Eliminates array index confusion like `[south, east, north, west]`
/// vs `[north, south, east, west]` by using explicit field names.
///
/// # Example
///
/// ```
/// use dgcom::types::SideBoundaries;
/// use dgcom::mesh::BoundaryTag;
///
/// let bcs = SideBoundaries::new(
///     BoundaryTag::Wall,   // south
///     BoundaryTag::Open,   // east
///     BoundaryTag::Wall,   // north
///     BoundaryTag::River,  // west
/// );
///
/// assert_eq!(bcs.south, BoundaryTag::Wall);
/// assert_eq!(bcs.east, BoundaryTag::Open);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SideBoundaries<T> {
    /// South boundary (y = y_min)
    pub south: T,
    /// East boundary (x = x_max)
    pub east: T,
    /// North boundary (y = y_max)
    pub north: T,
    /// West boundary (x = x_min)
    pub west: T,
}

impl<T> SideBoundaries<T> {
    /// Create new side boundaries with explicit named values.
    ///
    /// Order: south, east, north, west (counterclockwise from bottom)
    pub fn new(south: T, east: T, north: T, west: T) -> Self {
        Self {
            south,
            east,
            north,
            west,
        }
    }

    /// Create with the same value on all sides.
    pub fn uniform(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            south: value.clone(),
            east: value.clone(),
            north: value.clone(),
            west: value,
        }
    }

    /// Map a function over all sides.
    pub fn map<U, F>(self, mut f: F) -> SideBoundaries<U>
    where
        F: FnMut(T) -> U,
    {
        SideBoundaries {
            south: f(self.south),
            east: f(self.east),
            north: f(self.north),
            west: f(self.west),
        }
    }

    /// Get a reference to a side by index.
    ///
    /// Index convention: 0=south, 1=east, 2=north, 3=west
    pub fn get(&self, index: usize) -> Option<&T> {
        match index {
            0 => Some(&self.south),
            1 => Some(&self.east),
            2 => Some(&self.north),
            3 => Some(&self.west),
            _ => None,
        }
    }

    /// Convert to array [south, east, north, west].
    pub fn to_array(self) -> [T; 4] {
        [self.south, self.east, self.north, self.west]
    }

    /// Create from array [south, east, north, west].
    pub fn from_array([south, east, north, west]: [T; 4]) -> Self {
        Self {
            south,
            east,
            north,
            west,
        }
    }

    /// Iterate over sides in order: south, east, north, west.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.south, &self.east, &self.north, &self.west].into_iter()
    }
}

impl<T: Default> Default for SideBoundaries<T> {
    fn default() -> Self {
        Self {
            south: T::default(),
            east: T::default(),
            north: T::default(),
            west: T::default(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for SideBoundaries<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S:{} E:{} N:{} W:{}",
            self.south, self.east, self.north, self.west
        )
    }
}

impl<T> From<[T; 4]> for SideBoundaries<T> {
    fn from(arr: [T; 4]) -> Self {
        Self::from_array(arr)
    }
}

impl<T> From<SideBoundaries<T>> for [T; 4] {
    fn from(sides: SideBoundaries<T>) -> Self {
        sides.to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let sides = SideBoundaries::new(1, 2, 3, 4);
        assert_eq!(sides.south, 1);
        assert_eq!(sides.east, 2);
        assert_eq!(sides.north, 3);
        assert_eq!(sides.west, 4);
    }

    #[test]
    fn test_uniform() {
        let sides = SideBoundaries::uniform(42);
        assert_eq!(sides.south, 42);
        assert_eq!(sides.east, 42);
        assert_eq!(sides.north, 42);
        assert_eq!(sides.west, 42);
    }

    #[test]
    fn test_map() {
        let sides = SideBoundaries::new(1, 2, 3, 4);
        let doubled = sides.map(|x| x * 2);
        assert_eq!(doubled.south, 2);
        assert_eq!(doubled.east, 4);
        assert_eq!(doubled.north, 6);
        assert_eq!(doubled.west, 8);
    }

    #[test]
    fn test_get() {
        let sides = SideBoundaries::new(1, 2, 3, 4);
        assert_eq!(sides.get(0), Some(&1));
        assert_eq!(sides.get(1), Some(&2));
        assert_eq!(sides.get(2), Some(&3));
        assert_eq!(sides.get(3), Some(&4));
        assert_eq!(sides.get(4), None);
    }

    #[test]
    fn test_array_conversion() {
        let sides = SideBoundaries::new(1, 2, 3, 4);
        let arr = sides.to_array();
        assert_eq!(arr, [1, 2, 3, 4]);

        let back: SideBoundaries<i32> = arr.into();
        assert_eq!(back.south, 1);
    }

    #[test]
    fn test_iter() {
        let sides = SideBoundaries::new(1, 2, 3, 4);
        let collected: Vec<_> = sides.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3, 4]);
    }
}
