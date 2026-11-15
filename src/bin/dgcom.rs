//! `dgcom`: run a case file on the CPU (`dgcom::case::cli`; `gpu/`'s
//! `dgcom-gpu` adds its CUDA backend for `[compute] device`).

use std::process::ExitCode;

fn main() -> ExitCode {
    dgcom::case::cli::main(std::env::args().skip(1).collect(), None)
}
