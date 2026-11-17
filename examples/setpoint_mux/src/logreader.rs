//! Log export CLI for the demo (`extract-copperlists`, `fsck`, ...).

pub mod tasks;

use cu29::prelude::*;
use cu29_export::run_cli;

gen_cumsgs!("copperconfig.ron");

fn main() {
    // A copperlist holds 1024-point trajectories; decoding one needs more
    // than the default main-thread stack.
    let reader = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_cli::<CuMsgs>);
    let result = match reader {
        Ok(handle) => handle
            .join()
            .unwrap_or_else(|_| Err(CuError::from("logreader panicked"))),
        Err(e) => Err(CuError::new_with_cause("cannot spawn logreader thread", e)),
    };
    if let Err(e) = result {
        eprintln!("setpoint-mux-logreader: {e}");
        std::process::exit(1);
    }
}
