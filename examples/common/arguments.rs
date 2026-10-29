//! `key=value` command-line options that remember which keys were read, so a
//! parser can refuse the ones it never asked for: a misspelt key, or one a
//! binary built before it existed does not know (a farm sweep with `site=`
//! once ran as plain tides that way, `notes/tech-debt.md` P6).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// The command line's `key=value` options (a bare `key` has an empty value).
pub struct Arguments {
    values: HashMap<String, String>,
    read: RefCell<HashSet<String>>,
}

impl Arguments {
    /// The options of this process's command line.
    pub fn from_env() -> Self {
        Self::new(std::env::args().skip(1))
    }

    pub fn new(args: impl IntoIterator<Item = String>) -> Self {
        let values = args
            .into_iter()
            .map(|a| match a.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => (a, String::new()),
            })
            .collect();
        Self {
            values,
            read: RefCell::default(),
        }
    }

    /// The value of `key` (`Some("")` for a flag), which counts as read
    /// whether it is given or not.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.read.borrow_mut().insert(key.to_string());
        self.values.get(key)
    }

    /// Refuse the options no `get` asked for, once the parser is done; an
    /// option that only counts with another (the farm's without `site=`) is
    /// one of them.
    pub fn refuse_unread(&self) -> Result<(), String> {
        let read = self.read.borrow();
        let mut unread: Vec<&str> = self
            .values
            .keys()
            .filter(|k| !read.contains(k.as_str()))
            .map(String::as_str)
            .collect();
        if unread.is_empty() {
            return Ok(());
        }
        unread.sort_unstable();
        Err(format!(
            "unknown or unused option(s): {} (an option this binary does not \
             know, misspelt, or one that needs another option)",
            unread.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(args: &[&str]) -> Arguments {
        Arguments::new(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn options_read_or_asked_for_pass() {
        let args = arguments(&["nx=60", "wind"]);
        assert_eq!(args.get("nx").map(String::as_str), Some("60"));
        assert_eq!(args.get("wind").map(String::as_str), Some(""));
        assert_eq!(args.get("ny"), None);
        assert_eq!(args.refuse_unread(), Ok(()));
    }

    #[test]
    fn options_nothing_read_are_refused() {
        let args = arguments(&["nx=60", "site=farm.txt", "nu=0.1", "flag"]);
        args.get("nx");
        let error = args.refuse_unread().unwrap_err();
        assert!(error.contains("flag, nu, site"), "{error}");
    }
}
