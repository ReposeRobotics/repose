//! Zero-allocation gate for [`CuActionChunkTask`] feeding VLA action tasks
//! (`CuCartesianActionTask` + `CuGripperActionTask`) over a realistic `H = 50,
//! ACT = 7` horizon.
//!
//! Unlike `tract` inference (bounded in `tests/alloc.rs`), chunk ingestion,
//! exponential temporal ensembling, receding-horizon stepping, differential IK,
//! and gripper hysteresis are on the high-rate control path and must perform
//! **zero** heap allocations per cycle.

use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use repose_io::CuGripperCommandPayload;
use repose_kinematics::CuChainSpec;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotPreset;
use repose_kinematics::CuRobotStatus;
use repose_policy_rl::CuActionChunkMode;
use repose_policy_rl::CuActionChunkParams;
use repose_policy_rl::CuActionChunkTask;
use repose_policy_rl::CuCartesianActionParams;
use repose_policy_rl::CuCartesianActionReport;
use repose_policy_rl::CuCartesianActionTask;
use repose_policy_rl::CuGripperActionParams;
use repose_policy_rl::CuGripperActionTask;
use repose_policy_rl::CuGripperCloseWhen;
use repose_policy_rl::CuPolicyAction;
use repose_policy_rl::CuPolicyActionChunk;
use repose_policy_rl::CuPolicyStatus;
use repose_policy_rl::CuRotationEncoding;
use repose_trajectory::CuTrajectoryFollowOutput;
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

fn pos(rad: f64) -> CuJointPosition {
    CuJointPosition::Revolute(Angle::new::<radian>(rad))
}

fn vel(rad_s: f64) -> CuJointVelocity {
    CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(rad_s))
}

#[test]
fn chunk_ensembler_and_vla_action_tasks_never_allocate_on_hot_path() {
    let mut chunker = CuActionChunkTask::<50, 7>::from_params(CuActionChunkParams {
        mode: CuActionChunkMode::TemporalEnsemble { decay: 0.01 },
        step_decimation: 2,
        execution_horizon: Some(32),
    })
    .unwrap();
    let mut arm = CuCartesianActionTask::<6, 7>::from_params(CuCartesianActionParams {
        chain: Some(CuChainSpec::Preset(CuRobotPreset::IgusRebel)),
        period: Some(Time::new::<second>(0.02)),
        rotation: CuRotationEncoding::Rpy,
        translation_scale: Length::new::<meter>(0.005),
        rotation_scale: Angle::new::<radian>(0.02),
        ..CuCartesianActionParams::default()
    })
    .unwrap();
    let mut gripper = CuGripperActionTask::<7>::from_params(CuGripperActionParams {
        offset: 6,
        threshold: 0.5,
        close_when: CuGripperCloseWhen::Below,
        hysteresis: 0.1,
    })
    .unwrap();

    let (ctx, _clock) = CuContext::new_mock_clock();
    let fresh_msg = CuMsg::new(Some(CuPolicyActionChunk::<50, 7> {
        values: core::array::from_fn(|s| {
            let phase = s as f32 * 0.02;
            [
                0.2,
                -0.1,
                0.15,
                0.05,
                -0.05,
                0.1,
                if s < 25 { 1.0 } else { phase },
            ]
        }),
        status: CuPolicyStatus::Fresh,
    }));
    let empty_msg = CuMsg::<CuPolicyActionChunk<50, 7>>::new(None);
    let state_msg = CuMsg::new(Some(CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [0.0, 0.5, 1.0, 0.0, 0.8, 0.0].map(pos),
        velocities: [vel(0.0); 6],
    }));

    let mut action_msg = CuMsg::<CuPolicyAction<7>>::default();
    let mut arm_out = (
        CuMsg::<CuTrajectoryFollowOutput<6>>::default(),
        CuMsg::<CuCartesianActionReport>::default(),
    );
    let mut grip_out = CuMsg::<CuGripperCommandPayload>::default();

    for cycle in 0..32usize {
        let in_msg = if cycle.is_multiple_of(8) {
            &fresh_msg
        } else {
            &empty_msg
        };
        chunker.process(&ctx, in_msg, &mut action_msg).unwrap();
        arm.process(&ctx, &(&action_msg, &state_msg), &mut arm_out)
            .unwrap();
        gripper.process(&ctx, &action_msg, &mut grip_out).unwrap();
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);
    for cycle in 0..1000usize {
        let in_msg = if cycle.is_multiple_of(8) {
            &fresh_msg
        } else {
            &empty_msg
        };
        chunker.process(&ctx, in_msg, &mut action_msg).unwrap();
        arm.process(&ctx, &(&action_msg, &state_msg), &mut arm_out)
            .unwrap();
        gripper.process(&ctx, &action_msg, &mut grip_out).unwrap();
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        allocations, 0,
        "expected 0 heap allocations over 1000 RT cycles, measured {allocations}"
    );
}
