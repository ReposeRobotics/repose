//! Virtual workspace fences and node parameters for [`CuSafetyShieldTask`](crate::CuSafetyShieldTask).

use crate::report::CuSafetyWitness;
use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use cu29::units::si::velocity::meter_per_second;
use repose_collision::CuCollisionModelSpec;
use repose_collision::CuRobotCollisionModel;
use repose_kinematics::CuChainSpec;
use repose_kinematics::CuKinematicChain;
use serde::Deserialize;
use serde::Serialize;

/// Default [`CuSafetyShieldParams::near_miss_margin`]: 20 mm.
pub const DEFAULT_NEAR_MISS_MARGIN_M: f64 = 0.02;

fn default_near_miss_margin() -> Length {
    Length::new::<meter>(DEFAULT_NEAR_MISS_MARGIN_M)
}

/// Axis-aligned keep-in box for the tool centre point (TCP) in the robot base frame.
///
/// Every coordinate must lie in `min[axis]..=max[axis]`. Useful for enforcing a
/// table floor (`min[2] = 0.02 m`) and reach cage without modelling walls as
/// collision primitives.
///
/// ```ron
/// "workspace": {"min": [-0.6, -0.6, 0.02], "max": [0.6, 0.6, 0.9]}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[serde(deny_unknown_fields)]
pub struct CuWorkspaceBounds {
    /// Minimum TCP coordinates `[x, y, z]` in base frame.
    pub min: [Length; 3],
    /// Maximum TCP coordinates `[x, y, z]` in base frame.
    pub max: [Length; 3],
}

impl CuWorkspaceBounds {
    /// Creates an axis-aligned TCP keep-in box, rejecting non-finite bounds or `min > max`.
    pub fn new(min: [Length; 3], max: [Length; 3]) -> CuResult<Self> {
        let bounds = Self { min, max };
        bounds.validate()?;
        Ok(bounds)
    }

    /// Checks that every bound is finite and `min[axis] <= max[axis]`.
    pub fn validate(&self) -> CuResult<()> {
        for axis in 0..3 {
            let lo = self.min[axis].get::<meter>();
            let hi = self.max[axis].get::<meter>();
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(CuError::from(
                    "repose-safety: workspace bounds must be finite with min <= max",
                ));
            }
        }
        Ok(())
    }

    /// Evaluates the minimum signed distance from `point_m` to the six box faces
    /// (positive inside, negative outside) and the tightest face witness.
    pub(crate) fn evaluate_point(&self, point_m: [f64; 3]) -> (f64, CuSafetyWitness) {
        let mut min_dist = f64::INFINITY;
        let mut witness = CuSafetyWitness::None;
        for (axis, &coord) in point_m.iter().enumerate() {
            let lo = self.min[axis].get::<meter>();
            let hi = self.max[axis].get::<meter>();
            let d_lo = coord - lo;
            if d_lo < min_dist || (witness == CuSafetyWitness::None && !d_lo.is_nan()) {
                min_dist = d_lo;
                witness = CuSafetyWitness::WorkspaceAxis {
                    axis: axis as u8,
                    upper: false,
                };
            }
            let d_hi = hi - coord;
            if d_hi < min_dist {
                min_dist = d_hi;
                witness = CuSafetyWitness::WorkspaceAxis {
                    axis: axis as u8,
                    upper: true,
                };
            }
        }
        (min_dist, witness)
    }
}

/// Parameters (node `config:`) of [`CuSafetyShieldTask`](crate::CuSafetyShieldTask).
///
/// ```ron
/// config: {
///     "chain": "igus_rebel",
///     "collision_model": "igus_rebel",
///     "workspace": {"min": [-0.6, -0.6, 0.02], "max": [0.6, 0.6, 0.9]},
///     "braking_horizon": 0.1,
///     "near_miss_margin": 0.02,
///     "max_tcp_speed": 1.0,
///     "require_scene": false,
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuSafetyShieldParams<const N: usize, const M: usize = N> {
    /// Required robot chain: a preset name (`"igus_rebel"`) or an inline chain.
    #[serde(default)]
    #[reflect(ignore)]
    pub chain: Option<CuChainSpec<N>>,
    /// Required collision model: a preset name (`"igus_rebel"`) or an inline model.
    #[serde(default)]
    #[reflect(ignore)]
    pub collision_model: Option<CuCollisionModelSpec<M>>,
    /// Optional TCP keep-in box in base frame (`None` disables fence check).
    #[serde(default)]
    pub workspace: Option<CuWorkspaceBounds>,
    /// Optional lookahead horizon (`t_brake > 0`); checks linearly extrapolated
    /// stopping configuration `clamp_to_limits(q + v * t_brake)`.
    #[serde(default)]
    pub braking_horizon: Option<Time>,
    /// Non-negative distance threshold above `model.safety_margin()` that flags
    /// `near_miss: true` in [`CuSafetyReport`](crate::CuSafetyReport) while still `Clear`.
    #[serde(default = "default_near_miss_margin")]
    pub near_miss_margin: Length,
    /// Optional Cartesian TCP linear speed limit computed via `geometric_jacobian`.
    #[serde(default)]
    pub max_tcp_speed: Option<Velocity>,
    /// When `true`, a missing [`CuPlanningScene`](repose_collision::CuPlanningScene)
    /// payload triggers `HeldMissingScene`.
    #[serde(default)]
    pub require_scene: bool,
}

impl<const N: usize, const M: usize> Default for CuSafetyShieldParams<N, M> {
    fn default() -> Self {
        Self {
            chain: None,
            collision_model: None,
            workspace: None,
            braking_horizon: None,
            near_miss_margin: default_near_miss_margin(),
            max_tcp_speed: None,
            require_scene: false,
        }
    }
}

impl<const N: usize, const M: usize> CuSafetyShieldParams<N, M> {
    /// Resolves and validates the configured kinematic chain.
    pub fn resolve_chain(&self) -> CuResult<CuKinematicChain<N>> {
        self.chain
            .as_ref()
            .ok_or_else(|| CuError::from("repose-safety: missing required `chain` configuration"))?
            .resolve()
    }

    /// Resolves and validates the configured robot collision model.
    pub fn resolve_collision_model(&self) -> CuResult<CuRobotCollisionModel<N, M>> {
        self.collision_model
            .as_ref()
            .ok_or_else(|| {
                CuError::from("repose-safety: missing required `collision_model` configuration")
            })?
            .resolve()
    }

    /// Validates all parameters, including required `chain` and `collision_model`.
    pub fn validate(&self) -> CuResult<()> {
        self.resolve_chain()?;
        self.resolve_collision_model()?;
        if let Some(workspace) = &self.workspace {
            workspace.validate()?;
        }
        if let Some(horizon) = self.braking_horizon {
            let s = horizon.get::<second>();
            if !s.is_finite() || s <= 0.0 {
                return Err(CuError::from(
                    "repose-safety: `braking_horizon` must be positive and finite",
                ));
            }
        }
        let margin = self.near_miss_margin.get::<meter>();
        if !margin.is_finite() || margin < 0.0 {
            return Err(CuError::from(
                "repose-safety: `near_miss_margin` must be finite and non-negative",
            ));
        }
        if let Some(max_speed) = self.max_tcp_speed {
            let v = max_speed.get::<meter_per_second>();
            if !v.is_finite() || v <= 0.0 {
                return Err(CuError::from(
                    "repose-safety: `max_tcp_speed` must be positive and finite",
                ));
            }
        }
        Ok(())
    }
}
