//! Tool-frame path constraints enforced along the whole planned path.
//!
//! A planned motion is otherwise only checked for joint limits and collisions.
//! Users carrying a cup, pointing a camera, or holding a welding torch need the
//! tool to stay within bounds for the whole move, not just at the goal. A
//! [`CuPathConstraintSet`] carries up to [`MAX_PATH_CONSTRAINTS`] constraints,
//! all of which must hold (conjunction):
//!
//! - [`CuOrientationConstraint`]: a chosen tool-frame axis stays within a cone
//!   angle of a reference direction in the base frame ("keep the tool
//!   upright").
//! - [`CuPositionRegionConstraint`]: the tool centre point stays inside an
//!   axis-aligned box in the base frame.
//!
//! [`CuSceneValidity`](crate::CuSceneValidity) optionally carries a set and
//! rejects violating configurations, so every planner honours constraints
//! without per-planner logic. Enforcement is rejection of violating samples at
//! the planner's discrete checking resolution — the same sampling that checks
//! collisions, not a continuous guarantee. Very tight tolerances shrink the
//! free space the planners sample from, which makes planning slow or lets jobs
//! fail with [`Exhausted`](crate::CuPlanStatus::Exhausted): a cone of a few
//! degrees or a box of a few millimetres rejects almost every random sample,
//! and the gradient optimizer cannot steer toward satisfaction at all (its
//! cost only sees collisions and smoothness). Prefer the loosest tolerance the
//! task allows.
//!
//! Keep the tool Z axis within 10 degrees of world up for the whole move:
//!
//! ```rust
//! use cu29::units::si::angle::degree;
//! use cu29::units::si::angle::radian;
//! use cu29::units::si::f64::Angle;
//! use cu29::units::si::f64::Length;
//! use cu29::units::si::length::meter;
//! use cu_spatial_payloads::Transform3D;
//! use repose_motion_planning::CuOrientationConstraint;
//! use repose_motion_planning::CuPathConstraint;
//! use repose_motion_planning::CuPathConstraintSet;
//! use repose_motion_planning::CuToolAxis;
//!
//! let mut set = CuPathConstraintSet::new();
//! set.push(CuPathConstraint::Orientation(
//!     CuOrientationConstraint::new(
//!         CuToolAxis::Z,
//!         [0.0, 0.0, 1.0],
//!         Angle::new::<degree>(10.0),
//!     )
//!     .unwrap(),
//! ))
//! .unwrap();
//!
//! // The identity tool pose points straight up: inside the cone.
//! assert!(set.is_satisfied_by_pose(&Transform3D::<f64>::identity()));
//! // Tipped 90 degrees about X: outside the cone.
//! let tipped = Transform3D::<f64>::from_translation_euler_xyz(
//!     [Length::new::<meter>(0.0); 3],
//!     [
//!         Angle::new::<degree>(90.0),
//!         Angle::new::<radian>(0.0),
//!         Angle::new::<radian>(0.0),
//!     ],
//! );
//! assert!(!set.is_satisfied_by_pose(&tipped));
//!
//! // The same set from a planner task's `"constraints"` RON key:
//! let config: cu29::prelude::ComponentConfig = ron::from_str(
//!     r#"#![enable(unwrap_newtypes)] {"constraints": [{"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": 0.1745}}]}"#,
//! )
//! .unwrap();
//! let ron: CuPathConstraintSet = config.get_value("constraints").unwrap().unwrap();
//! assert_eq!(ron.len(), 1);
//! ```

use bincode::Decode;
use bincode::Encode;
use cu_spatial_payloads::Transform3D;
use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use serde::Deserialize;
use serde::Serialize;

/// Fixed capacity of constraints in a [`CuPathConstraintSet`].
pub const MAX_PATH_CONSTRAINTS: usize = 4;

/// A tool-frame axis, for [`CuOrientationConstraint`].
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuToolAxis {
    /// Tool X axis (first column of the tool rotation matrix).
    X,
    /// Tool Y axis (second column of the tool rotation matrix).
    Y,
    /// Tool Z axis (third column of the tool rotation matrix).
    #[default]
    Z,
}

impl CuToolAxis {
    /// Column index of this axis in a row-major rotation matrix.
    const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Constrains a tool-frame axis to a cone around a base-frame direction.
///
/// The axis [`tool_axis`](Self::tool_axis), expressed in the base frame,
/// must stay within [`tolerance`](Self::tolerance) of the unit-free
/// [`reference`](Self::reference) direction (any finite non-zero vector; only
/// its direction matters).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(deny_unknown_fields)]
pub struct CuOrientationConstraint {
    /// Tool-frame axis the cone applies to.
    pub tool_axis: CuToolAxis,
    /// Reference direction in the base frame (dimensionless, finite, non-zero).
    pub reference: [f64; 3],
    /// Cone half-angle around the reference direction.
    pub tolerance: Angle,
}

impl CuOrientationConstraint {
    /// Creates a cone constraint, rejecting non-finite or zero reference
    /// directions and tolerances outside `0..=pi`.
    pub fn new(tool_axis: CuToolAxis, reference: [f64; 3], tolerance: Angle) -> CuResult<Self> {
        let constraint = Self {
            tool_axis,
            reference,
            tolerance,
        };
        constraint.validate()?;
        Ok(constraint)
    }

    /// Checks that the reference direction is finite and non-zero and the
    /// tolerance is a finite angle in `0..=pi`.
    pub fn validate(&self) -> CuResult<()> {
        if !self.reference.iter().all(|v| v.is_finite()) {
            return Err(CuError::from(
                "repose-motion-planning: orientation constraint reference must be finite",
            ));
        }
        let norm_squared = self.reference[0] * self.reference[0]
            + self.reference[1] * self.reference[1]
            + self.reference[2] * self.reference[2];
        if !norm_squared.is_finite() || norm_squared <= 0.0 {
            return Err(CuError::from(
                "repose-motion-planning: orientation constraint reference must be non-zero",
            ));
        }
        let tolerance = self.tolerance.get::<radian>();
        if !tolerance.is_finite() || tolerance < 0.0 || tolerance > core::f64::consts::PI {
            return Err(CuError::from(
                "repose-motion-planning: orientation constraint tolerance must be in 0..=pi",
            ));
        }
        Ok(())
    }

    /// `true` when the tool axis of `tcp` lies within the cone.
    pub fn is_satisfied_by_pose(&self, tcp: &Transform3D<f64>) -> bool {
        let tolerance = self.tolerance.get::<radian>();
        if !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let reference_norm = libm::sqrt(
            self.reference[0] * self.reference[0]
                + self.reference[1] * self.reference[1]
                + self.reference[2] * self.reference[2],
        );
        if !reference_norm.is_finite() || reference_norm <= 0.0 {
            return false;
        }
        // A pi cone accepts every direction; skip the cosine comparison whose
        // floating-point boundary could otherwise reject antiparallel axes.
        if tolerance >= core::f64::consts::PI {
            return true;
        }
        let rotation = tcp.rotation();
        let column = self.tool_axis.index();
        let axis = [
            rotation[0][column],
            rotation[1][column],
            rotation[2][column],
        ];
        let axis_norm = libm::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        if !axis_norm.is_finite() || axis_norm <= 0.0 {
            return false;
        }
        let dot = (axis[0] * self.reference[0]
            + axis[1] * self.reference[1]
            + axis[2] * self.reference[2])
            / (axis_norm * reference_norm);
        dot.is_finite() && dot >= libm::cos(tolerance)
    }
}

/// Constrains the tool centre point to an axis-aligned box in the base frame.
///
/// The translation of the tool pose must satisfy `min <= tcp <= max`
/// component-wise (bounds inclusive).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(deny_unknown_fields)]
pub struct CuPositionRegionConstraint {
    /// Lower box corner in the base frame.
    pub min: [Length; 3],
    /// Upper box corner in the base frame.
    pub max: [Length; 3],
}

impl CuPositionRegionConstraint {
    /// Creates a box constraint, rejecting non-finite bounds and corners with
    /// `min > max` in any component.
    pub fn new(min: [Length; 3], max: [Length; 3]) -> CuResult<Self> {
        let constraint = Self { min, max };
        constraint.validate()?;
        Ok(constraint)
    }

    /// Checks that both corners are finite with `min <= max` component-wise.
    pub fn validate(&self) -> CuResult<()> {
        for i in 0..3 {
            let lo = self.min[i].get::<meter>();
            let hi = self.max[i].get::<meter>();
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(CuError::from(
                    "repose-motion-planning: position region must have finite bounds with min <= max",
                ));
            }
        }
        Ok(())
    }

    /// `true` when the translation of `tcp` lies inside the box.
    pub fn is_satisfied_by_pose(&self, tcp: &Transform3D<f64>) -> bool {
        let translation = tcp.translation();
        (0..3).all(|i| translation[i] >= self.min[i] && translation[i] <= self.max[i])
    }
}

/// One tool-frame path constraint: an orientation cone or a position box.
#[derive(Debug, Clone, Copy, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum CuPathConstraint {
    /// A tool-frame axis stays within a cone around a base-frame direction.
    Orientation(CuOrientationConstraint),
    /// The tool centre point stays inside a base-frame box.
    PositionRegion(CuPositionRegionConstraint),
}

impl CuPathConstraint {
    /// Checks the constraint parameters.
    pub fn validate(&self) -> CuResult<()> {
        match self {
            Self::Orientation(constraint) => constraint.validate(),
            Self::PositionRegion(constraint) => constraint.validate(),
        }
    }

    /// `true` when the tool pose `tcp` satisfies this constraint.
    pub fn is_satisfied_by_pose(&self, tcp: &Transform3D<f64>) -> bool {
        match self {
            Self::Orientation(constraint) => constraint.is_satisfied_by_pose(tcp),
            Self::PositionRegion(constraint) => constraint.is_satisfied_by_pose(tcp),
        }
    }
}

/// Fixed-capacity set of path constraints, all of which must hold.
///
/// Holds up to [`MAX_PATH_CONSTRAINTS`] constraints; [`push`](Self::push)
/// validates each one. Serializes as a list, so a planner task's RON config
/// carries a `"constraints"` key next to `"params"`:
///
/// ```ron
/// "constraints": [
///     {"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": 0.17}},
///     {"position_region": {"min": [0.1, -0.5, 0.0], "max": [0.6, 0.5, 0.8]}},
/// ],
/// ```
#[derive(Default, Debug, Clone, Copy, PartialEq, Encode, Decode, Reflect)]
pub struct CuPathConstraintSet {
    constraints: [Option<CuPathConstraint>; MAX_PATH_CONSTRAINTS],
}

impl CuPathConstraintSet {
    /// An empty set, satisfied by every pose.
    pub const fn new() -> Self {
        Self {
            constraints: [None; MAX_PATH_CONSTRAINTS],
        }
    }

    /// Number of constraints in the set.
    pub fn len(&self) -> usize {
        self.constraints.iter().filter(|c| c.is_some()).count()
    }

    /// `true` when the set holds no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.iter().all(|c| c.is_none())
    }

    /// The constraint at `index`, or `None` past the end.
    pub fn get(&self, index: usize) -> Option<&CuPathConstraint> {
        self.constraints.get(index)?.as_ref()
    }

    /// Iterates over the constraints in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &CuPathConstraint> + '_ {
        self.constraints.iter().filter_map(|c| c.as_ref())
    }

    /// Appends `constraint` after validating it; fails when invalid or full.
    pub fn push(&mut self, constraint: CuPathConstraint) -> CuResult<()> {
        constraint.validate()?;
        let Some(slot) = self.constraints.iter_mut().find(|c| c.is_none()) else {
            return Err(CuError::from(
                "repose-motion-planning: too many path constraints",
            ));
        };
        *slot = Some(constraint);
        Ok(())
    }

    /// Removes every constraint.
    pub fn clear(&mut self) {
        self.constraints = [None; MAX_PATH_CONSTRAINTS];
    }

    /// Checks every constraint in the set.
    pub fn validate(&self) -> CuResult<()> {
        for constraint in self.iter() {
            constraint.validate()?;
        }
        Ok(())
    }

    /// `true` when the tool pose `tcp` satisfies every constraint in the set.
    pub fn is_satisfied_by_pose(&self, tcp: &Transform3D<f64>) -> bool {
        self.iter().all(|c| c.is_satisfied_by_pose(tcp))
    }
}

impl Serialize for CuPathConstraintSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for constraint in self.iter() {
            seq.serialize_element(constraint)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for CuPathConstraintSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::SeqAccess;
        use serde::de::Visitor;
        struct SetVisitor;
        impl<'de> Visitor<'de> for SetVisitor {
            type Value = CuPathConstraintSet;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a list of at most 4 path constraints")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut set = CuPathConstraintSet::new();
                while let Some(constraint) = seq.next_element::<CuPathConstraint>()? {
                    set.push(constraint).map_err(serde::de::Error::custom)?;
                }
                Ok(set)
            }
        }
        deserializer.deserialize_seq(SetVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::degree;

    fn upright(tolerance_deg: f64) -> CuPathConstraint {
        CuPathConstraint::Orientation(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 1.0],
                Angle::new::<degree>(tolerance_deg),
            )
            .unwrap(),
        )
    }

    fn tilted_about_x(degrees: f64) -> Transform3D<f64> {
        Transform3D::<f64>::from_translation_euler_xyz(
            [Length::new::<meter>(0.0); 3],
            [
                Angle::new::<degree>(degrees),
                Angle::new::<radian>(0.0),
                Angle::new::<radian>(0.0),
            ],
        )
    }

    #[test]
    fn test_orientation_cone_accepts_inside_and_rejects_outside() {
        let constraint = upright(10.0);
        assert!(constraint.is_satisfied_by_pose(&Transform3D::<f64>::identity()));
        assert!(constraint.is_satisfied_by_pose(&tilted_about_x(9.0)));
        assert!(!constraint.is_satisfied_by_pose(&tilted_about_x(11.0)));
        assert!(!constraint.is_satisfied_by_pose(&tilted_about_x(90.0)));
    }

    #[test]
    fn test_orientation_tool_axis_selects_the_constrained_column() {
        let constraint = CuPathConstraint::Orientation(
            CuOrientationConstraint::new(
                CuToolAxis::X,
                [1.0, 0.0, 0.0],
                Angle::new::<degree>(10.0),
            )
            .unwrap(),
        );
        // A rotation about X leaves the X axis fixed: still satisfied.
        assert!(constraint.is_satisfied_by_pose(&tilted_about_x(45.0)));
        // A rotation about Z moves the X axis: violated.
        let yawed = Transform3D::<f64>::from_translation_euler_xyz(
            [Length::new::<meter>(0.0); 3],
            [
                Angle::new::<radian>(0.0),
                Angle::new::<radian>(0.0),
                Angle::new::<degree>(45.0),
            ],
        );
        assert!(!constraint.is_satisfied_by_pose(&yawed));
    }

    #[test]
    fn test_orientation_reference_direction_only_not_magnitude() {
        let unit = CuOrientationConstraint::new(
            CuToolAxis::Z,
            [0.0, 0.0, 1.0],
            Angle::new::<degree>(10.0),
        )
        .unwrap();
        let scaled = CuOrientationConstraint::new(
            CuToolAxis::Z,
            [0.0, 0.0, 5.0],
            Angle::new::<degree>(10.0),
        )
        .unwrap();
        for pose in [Transform3D::<f64>::identity(), tilted_about_x(11.0)] {
            assert_eq!(
                unit.is_satisfied_by_pose(&pose),
                scaled.is_satisfied_by_pose(&pose)
            );
        }
    }

    #[test]
    fn test_orientation_extreme_tolerances() {
        let exact =
            CuOrientationConstraint::new(CuToolAxis::Z, [0.0, 0.0, 1.0], Angle::new::<radian>(0.0))
                .unwrap();
        assert!(exact.is_satisfied_by_pose(&Transform3D::<f64>::identity()));
        assert!(!exact.is_satisfied_by_pose(&tilted_about_x(1.0)));
        let anything = CuOrientationConstraint::new(
            CuToolAxis::Z,
            [0.0, 0.0, 1.0],
            Angle::new::<radian>(core::f64::consts::PI),
        )
        .unwrap();
        assert!(anything.is_satisfied_by_pose(&tilted_about_x(180.0)));
    }

    #[test]
    fn test_position_region_accepts_inside_and_rejects_outside() {
        let constraint = CuPositionRegionConstraint::new(
            [0.1, -0.5, 0.0].map(Length::new::<meter>),
            [0.6, 0.5, 0.8].map(Length::new::<meter>),
        )
        .unwrap();
        let inside = Transform3D::<f64>::from_translation_euler_xyz(
            [0.3, 0.0, 0.4].map(Length::new::<meter>),
            [Angle::new::<radian>(0.0); 3],
        );
        assert!(constraint.is_satisfied_by_pose(&inside));
        // Bounds are inclusive.
        let corner = Transform3D::<f64>::from_translation_euler_xyz(
            [0.1, -0.5, 0.8].map(Length::new::<meter>),
            [Angle::new::<radian>(0.0); 3],
        );
        assert!(constraint.is_satisfied_by_pose(&corner));
        for outside in [[0.09, 0.0, 0.4], [0.3, 0.51, 0.4], [0.3, 0.0, 0.81]] {
            let pose = Transform3D::<f64>::from_translation_euler_xyz(
                outside.map(Length::new::<meter>),
                [Angle::new::<radian>(0.0); 3],
            );
            assert!(!constraint.is_satisfied_by_pose(&pose));
        }
    }

    #[test]
    fn test_invalid_parameters_are_rejected() {
        assert!(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 0.0],
                Angle::new::<degree>(10.0)
            )
            .is_err()
        );
        assert!(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, f64::NAN],
                Angle::new::<degree>(10.0)
            )
            .is_err()
        );
        assert!(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 1.0],
                Angle::new::<radian>(-0.1)
            )
            .is_err()
        );
        assert!(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 1.0],
                Angle::new::<radian>(core::f64::consts::PI + 0.1)
            )
            .is_err()
        );
        assert!(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 1.0],
                Angle::new::<radian>(f64::INFINITY)
            )
            .is_err()
        );
        let lo = [0.0; 3].map(Length::new::<meter>);
        let hi = [1.0; 3].map(Length::new::<meter>);
        assert!(CuPositionRegionConstraint::new(lo, hi).is_ok());
        assert!(CuPositionRegionConstraint::new(hi, lo).is_err());
        assert!(
            CuPositionRegionConstraint::new([f64::NAN, 0.0, 0.0].map(Length::new::<meter>), hi)
                .is_err()
        );
        // Struct literals bypass `new`, so `validate` rejects them too.
        assert!(
            CuPositionRegionConstraint { min: hi, max: lo }
                .validate()
                .is_err()
        );
        // A set only holds validated constraints.
        let mut set = CuPathConstraintSet::new();
        assert!(set.push(upright(10.0)).is_ok());
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
        assert!(
            set.push(CuPathConstraint::Orientation(CuOrientationConstraint {
                tool_axis: CuToolAxis::Z,
                reference: [0.0, 0.0, 0.0],
                tolerance: Angle::new::<degree>(10.0),
            }))
            .is_err()
        );
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_set_enforces_capacity_and_conjunction() {
        let mut set = CuPathConstraintSet::new();
        assert!(set.is_empty());
        for _ in 0..MAX_PATH_CONSTRAINTS {
            set.push(upright(90.0)).unwrap();
        }
        assert_eq!(set.len(), MAX_PATH_CONSTRAINTS);
        assert!(set.push(upright(90.0)).is_err());
        assert!(set.get(MAX_PATH_CONSTRAINTS).is_none());
        assert_eq!(set.iter().count(), MAX_PATH_CONSTRAINTS);
        set.clear();
        assert!(set.is_empty());
        assert!(set.is_satisfied_by_pose(&Transform3D::<f64>::identity()));

        // Every constraint must hold: the region passes but the cone fails.
        set.push(upright(10.0)).unwrap();
        set.push(CuPathConstraint::PositionRegion(
            CuPositionRegionConstraint::new(
                [-1.0; 3].map(Length::new::<meter>),
                [1.0; 3].map(Length::new::<meter>),
            )
            .unwrap(),
        ))
        .unwrap();
        assert!(set.is_satisfied_by_pose(&Transform3D::<f64>::identity()));
        assert!(!set.is_satisfied_by_pose(&tilted_about_x(45.0)));
    }

    #[test]
    fn test_set_ron_round_trip() {
        let mut set = CuPathConstraintSet::new();
        set.push(upright(10.0)).unwrap();
        set.push(CuPathConstraint::PositionRegion(
            CuPositionRegionConstraint::new(
                [0.1, -0.5, 0.0].map(Length::new::<meter>),
                [0.6, 0.5, 0.8].map(Length::new::<meter>),
            )
            .unwrap(),
        ))
        .unwrap();
        let ron = ron::to_string(&set).unwrap();
        assert_eq!(ron::from_str::<CuPathConstraintSet>(&ron).unwrap(), set);
        // Over capacity and invalid entries fail through the node-config path.
        let one = "{\"orientation\": {\"tool_axis\": \"z\", \"reference\": [0.0, 0.0, 1.0], \"tolerance\": 0.1}}";
        let full = format!("[{one}, {one}, {one}, {one}, {one}]");
        for constraints in [
            full,
            "[{\"orientation\": {\"tool_axis\": \"z\", \"reference\": [0.0, 0.0, 0.0], \"tolerance\": 0.1}}]"
                .into(),
            "[{\"position_region\": {\"min\": [1.0, 0.0, 0.0], \"max\": [0.0, 0.0, 0.0]}}]"
                .into(),
        ] {
            let config: ComponentConfig =
                ron::from_str(&format!("#![enable(unwrap_newtypes)] {{\"constraints\": {constraints}}}"))
                    .unwrap();
            assert!(
                config
                    .get_value::<CuPathConstraintSet>("constraints")
                    .is_err()
            );
        }
    }
}
