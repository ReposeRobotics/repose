//! Allocation gate: `tract` is the one documented exception to the
//! zero-allocation hot-path rule in this workspace, so the number of heap
//! allocations per `process` call is measured and bounded here instead of
//! assumed. The measured count is recorded in the crate docs.
//!
//! This is the only test in its binary: the counting allocator is global.

use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointPositionsPayload;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;
use repose_policy_rl::CuObservationTerm;
use repose_policy_rl::CuPolicyAction;
use repose_policy_rl::CuPolicyObservation;
use repose_policy_rl::CuPolicyParams;
use repose_policy_rl::CuPolicyTask;
use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Upper bound on heap allocations per `process` call (see crate docs).
/// Measured: 14 per call with `linear_24_6` (debug and release); the bound
/// leaves headroom for platform variance while catching regressions.
const MAX_ALLOCATIONS_PER_CALL: usize = 32;

fn pos(rad: f64) -> CuJointPosition {
    CuJointPosition::Revolute(Angle::new::<radian>(rad))
}

fn vel(rad_s: f64) -> CuJointVelocity {
    CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(rad_s))
}

#[test]
fn allocations_per_process_are_bounded() {
    let params = CuPolicyParams::<6, 24, 6> {
        model: format!(
            "{}/tests/fixtures/linear_24_6.onnx",
            env!("CARGO_MANIFEST_DIR")
        ),
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: 0.1 },
            CuObservationTerm::GoalError { scale: 1.0 },
            CuObservationTerm::PreviousAction,
        ],
        ..CuPolicyParams::default()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let (ctx, _clock) = CuContext::new_mock_clock();

    let state_msg = CuMsg::new(Some(CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map(pos),
        velocities: [vel(0.0); 6],
    }));
    let goal_msg = CuMsg::new(Some(CuJointPositionsPayload::<6> {
        positions: [pos(1.0); 6],
    }));
    let input = (&state_msg, &goal_msg);
    let mut output = (
        CuMsg::<CuPolicyObservation<24>>::default(),
        CuMsg::<CuPolicyAction<6>>::default(),
    );

    // Warm up: model caches and lazy allocations settle here, not in the count.
    for _ in 0..100 {
        task.process(&ctx, &input, &mut output).unwrap();
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);
    const CALLS: usize = 1000;
    for _ in 0..CALLS {
        task.process(&ctx, &input, &mut output).unwrap();
    }
    let total = ALLOCATIONS.load(Ordering::Relaxed);
    let per_call = total as f64 / CALLS as f64;
    println!("allocations: {total} over {CALLS} process calls ({per_call:.2} per call)");
    assert!(
        output.1.payload().is_some(),
        "the measured cycles must produce actions"
    );
    assert!(
        total <= MAX_ALLOCATIONS_PER_CALL * CALLS,
        "{per_call:.2} allocations per process call exceeds the bound of {MAX_ALLOCATIONS_PER_CALL}"
    );
}
