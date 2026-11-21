//! Log reader for `simplex_safety_demo`: dumps copperlists or runs `fsck`.
//!
//! ```text
//! cargo run --bin simplex_safety_logreader -- logs/simplex_v1.copper extract-copperlists -e jsonl
//! cargo run --bin simplex_safety_logreader -- logs/simplex_v1.copper fsck
//! ```

pub mod tasks;

use cu29::prelude::*;
use cu29_export::run_cli;

gen_cumsgs!("copperconfig.ron");

fn main() {
    // 1024-point trajectories make CuMsgs large; run with a generous stack.
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| run_cli::<CuMsgs>().expect("logreader failed"))
        .expect("cannot spawn logreader thread");
    handle.join().expect("logreader thread panicked");
}
