//! Rigid-body dynamics for serial chains: inertial parameters, inverse dynamics, and gravity compensation.
//!
//! [`CuChainInertia`] holds the per-link mass, centre of mass, and rotational
//! inertia of an `N`-DOF chain. [`CuKinematicChain::inverse_dynamics`] runs the
//! recursive Newton-Euler algorithm over those parameters, and
//! [`CuKinematicChain::gravity_compensation`] is the same recursion with zero
//! velocity and acceleration. Both write into caller-provided storage and
//! allocate nothing.
//!
//! Link frame `i` is the frame after joint `i`'s motion transform: the same
//! frame [`CuKinematicChain::forward_frames`] reports. Each [`CuLinkInertia`]
//! is expressed in its link frame, matching the URDF `<inertial>` convention
//! where the centre of mass and inertia tensor live in the child link frame.
//! Fixed joints carry no effort output ([`CuJointEffort::Fixed`]); prismatic
//! joints produce forces, revolute joints torques.

use crate::chain::CuJointType;
use crate::chain::CuKinematicChain;
use crate::joint::CuJointAcceleration;
use crate::joint::CuJointEffort;
use crate::joint::CuJointPosition;
use crate::joint::CuJointVelocity;
use cu29::prelude::*;
use cu29::units::si::acceleration::meter_per_second_squared;
use cu29::units::si::f64::Acceleration;
use cu29::units::si::f64::Force;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Mass;
use cu29::units::si::f64::MomentOfInertia;
use cu29::units::si::f64::Torque;
use cu29::units::si::force::newton;
use cu29::units::si::length::meter;
use cu29::units::si::mass::kilogram;
use cu29::units::si::moment_of_inertia::kilogram_square_meter;
use cu29::units::si::torque::newton_meter;
use serde::Deserialize;
use serde::Serialize;

/// Relative slack for the inertia tensor inequalities, absorbing float rounding of tabulated values.
const INERTIA_TOL: f64 = 1e-9;

/// Inertial parameters of one link: mass, centre of mass, and rotational inertia.
///
/// The centre of mass and the inertia tensor are expressed in the link frame
/// (the frame after the joint's motion transform). `inertia` holds the six
/// independent components of the symmetric tensor about the centre of mass in
/// kilogram square meters, ordered `[xx, yy, zz, xy, xz, yz]`. A point mass is
/// a positive `mass` with zero `inertia`; a massless link must be all zero.
///
/// The fields are public, so a value built or edited directly has not been
/// checked; [`Self::new`] and [`Self::validate`] do the checking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(deny_unknown_fields)]
pub struct CuLinkInertia {
    /// Link mass.
    pub mass: Mass,
    /// Centre of mass in the link frame.
    pub center_of_mass: [Length; 3],
    /// Rotational inertia about the centre of mass in the link frame (`[xx, yy, zz, xy, xz, yz]`).
    pub inertia: [MomentOfInertia; 6],
}

impl CuLinkInertia {
    /// Creates link inertial parameters, rejecting non-finite values, negative
    /// mass, and inertia tensors no rigid body could have.
    pub fn new(
        mass: Mass,
        center_of_mass: [Length; 3],
        inertia: [MomentOfInertia; 6],
    ) -> CuResult<Self> {
        let out = Self {
            mass,
            center_of_mass,
            inertia,
        };
        out.validate()?;
        Ok(out)
    }

    /// Checks finiteness, a non-negative mass, and physical plausibility of the
    /// inertia tensor. A massless link must have zero inertia.
    ///
    /// A rigid body's principal moments are non-negative and each is at most
    /// the sum of the other two. Both hold exactly when `(trace(I) / 2)·1 − I`
    /// is positive semi-definite, which is tested through its principal minors
    /// without an eigen-decomposition. A relative tolerance absorbs the
    /// rounding of tabulated values, so tensors on the boundary (point masses,
    /// thin rods, flat plates) pass.
    pub fn validate(&self) -> CuResult<()> {
        let mass = self.mass.get::<kilogram>();
        if !mass.is_finite() || mass < 0.0 {
            return Err(CuError::from("Link mass must be finite and non-negative"));
        }
        if self
            .center_of_mass
            .iter()
            .any(|c| !c.get::<meter>().is_finite())
        {
            return Err(CuError::from("Link centre of mass must be finite"));
        }
        let tensor = self.inertia.map(|v| v.get::<kilogram_square_meter>());
        if tensor.iter().any(|v| !v.is_finite()) {
            return Err(CuError::from("Link inertia components must be finite"));
        }
        let [xx, yy, zz, xy, xz, yz] = tensor;
        if mass == 0.0 {
            if tensor.iter().any(|v| *v != 0.0) {
                return Err(CuError::from("A massless link must have zero inertia"));
            }
            return Ok(());
        }
        if xx < 0.0 || yy < 0.0 || zz < 0.0 {
            return Err(CuError::from("Link inertia diagonal must be non-negative"));
        }
        // B = (trace / 2)·1 − I has the eigenvalues (λj + λk − λi) / 2 of the
        // principal moments λ, so B ⪰ 0 is the triangle inequality on the
        // principal moments, and it implies I ⪰ 0. B is positive semi-definite
        // when all seven of its principal minors are non-negative.
        let trace = xx + yy + zz;
        let half = trace / 2.0;
        let (bx, by, bz) = (half - xx, half - yy, half - zz);
        if bx < -INERTIA_TOL * trace || by < -INERTIA_TOL * trace || bz < -INERTIA_TOL * trace {
            return Err(CuError::from(
                "Link inertia violates the triangle inequalities",
            ));
        }
        let scale = trace / 3.0;
        let minor_tol = INERTIA_TOL * scale * scale;
        let det = bx * (by * bz - yz * yz) - xy * (xy * bz + yz * xz) - xz * (xy * yz + by * xz);
        if bx * by - xy * xy < -minor_tol
            || bx * bz - xz * xz < -minor_tol
            || by * bz - yz * yz < -minor_tol
            || det < -minor_tol * scale
        {
            return Err(CuError::from(
                "Link inertia principal moments violate the triangle inequalities",
            ));
        }
        Ok(())
    }
}

/// Compile-time sized inertial parameters of an $N$-DOF serial chain.
///
/// Entry `i` is the rigid body moved by joint `i` (the joint's child link,
/// including any links folded into it across fixed joints), expressed in link
/// frame `i`. Pairs with [`CuKinematicChain`] in
/// [`CuKinematicChain::inverse_dynamics`] and
/// [`CuKinematicChain::gravity_compensation`].
///
/// [`Self::new`] and deserialization both validate every link, so bad
/// parameters fail where they are loaded. Reflection can still write the links
/// in place, which is why [`CuKinematicChain::inverse_dynamics`] validates
/// again on every call instead of relying on this type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
#[serde(try_from = "CuChainInertiaUnchecked<N>")]
pub struct CuChainInertia<const N: usize> {
    #[serde(with = "crate::serde_arrays::link_inertia_array")]
    links: [CuLinkInertia; N],
}

/// Wire form of [`CuChainInertia`], validated on its way in.
#[derive(Deserialize)]
#[serde(rename = "CuChainInertia", deny_unknown_fields)]
struct CuChainInertiaUnchecked<const N: usize> {
    #[serde(with = "crate::serde_arrays::link_inertia_array")]
    links: [CuLinkInertia; N],
}

impl<const N: usize> TryFrom<CuChainInertiaUnchecked<N>> for CuChainInertia<N> {
    type Error = CuError;

    fn try_from(unchecked: CuChainInertiaUnchecked<N>) -> CuResult<Self> {
        Self::new(unchecked.links)
    }
}

impl<const N: usize> CuChainInertia<N> {
    /// Creates chain inertial parameters, validating every link.
    pub fn new(links: [CuLinkInertia; N]) -> CuResult<Self> {
        let out = Self { links };
        out.validate()?;
        Ok(out)
    }

    /// Checks every link's mass, centre of mass, and inertia tensor.
    pub fn validate(&self) -> CuResult<()> {
        let mut i = 0;
        while i < N {
            self.links[i].validate()?;
            i += 1;
        }
        Ok(())
    }

    /// Borrows the per-link inertial parameters.
    pub const fn links(&self) -> &[CuLinkInertia; N] {
        &self.links
    }
}

/// Borrowed inputs of [`CuKinematicChain::inverse_dynamics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuInverseDynamicsInput<'a, const N: usize> {
    /// Joint positions.
    pub positions: &'a [CuJointPosition; N],
    /// Joint velocities.
    pub velocities: &'a [CuJointVelocity; N],
    /// Joint accelerations.
    pub accelerations: &'a [CuJointAcceleration; N],
    /// Gravity acceleration vector in the base frame.
    pub gravity: [Acceleration; 3],
}

impl<'a, const N: usize> CuInverseDynamicsInput<'a, N> {
    /// Borrows the joint state and gravity for one inverse dynamics call.
    pub fn new(
        positions: &'a [CuJointPosition; N],
        velocities: &'a [CuJointVelocity; N],
        accelerations: &'a [CuJointAcceleration; N],
        gravity: [Acceleration; 3],
    ) -> Self {
        Self {
            positions,
            velocities,
            accelerations,
            gravity,
        }
    }
}

/// Standard Earth gravity in the base frame: `[0, 0, -9.80665]` m/s².
pub fn standard_gravity() -> [Acceleration; 3] {
    [
        Acceleration::new::<meter_per_second_squared>(0.0),
        Acceleration::new::<meter_per_second_squared>(0.0),
        Acceleration::new::<meter_per_second_squared>(-9.80665),
    ]
}

impl<const N: usize> CuKinematicChain<N> {
    /// Computes the joint efforts for the given motion by recursive Newton-Euler, in place.
    ///
    /// Given positions, velocities, accelerations, and gravity, writes the
    /// torque (revolute joints) or force (prismatic joints) each joint must
    /// exert into `out_efforts`. Fixed joints yield [`CuJointEffort::Fixed`].
    /// No tip load is modeled.
    ///
    /// Every input is checked on each call: `inertia` through
    /// [`CuChainInertia::validate`], because reflection can alter one after
    /// its validating constructor ran, and the joint state for finiteness and
    /// unit-kind agreement with the chain. Invalid input returns an error
    /// before `out_efforts` is written.
    pub fn inverse_dynamics(
        &self,
        inertia: &CuChainInertia<N>,
        input: &CuInverseDynamicsInput<N>,
        out_efforts: &mut [CuJointEffort; N],
    ) -> CuResult<()> {
        inertia.validate()?;
        let mut gravity = [0.0; 3];
        let mut k = 0;
        while k < 3 {
            gravity[k] = input.gravity[k].get::<meter_per_second_squared>();
            if !gravity[k].is_finite() {
                return Err(CuError::from("repose_kinematics: non-finite gravity"));
            }
            k += 1;
        }
        let joints = self.joints();
        let links = inertia.links();
        let mut qd = [0.0; N];
        let mut qdd = [0.0; N];
        let mut i = 0;
        while i < N {
            if !velocity_matches(input.positions[i], input.velocities[i])
                || !acceleration_matches(input.positions[i], input.accelerations[i])
            {
                return Err(CuError::from(
                    "Joint state units disagree across position, velocity, and acceleration",
                ));
            }
            qd[i] = input.velocities[i].si_value();
            qdd[i] = input.accelerations[i].si_value();
            if !qd[i].is_finite() || !qdd[i].is_finite() {
                return Err(CuError::from(
                    "repose_kinematics: non-finite joint velocity or acceleration",
                ));
            }
            i += 1;
        }

        let mut rot = [[[0.0; 3]; 3]; N];
        let mut org = [[0.0; 3]; N];
        let mut omega = [[0.0; 3]; N];
        let mut alpha = [[0.0; 3]; N];
        let mut acc = [[0.0; 3]; N];
        let mut acc_com = [[0.0; 3]; N];

        // Base frame: at rest, accelerating against gravity so weight appears
        // as an inertial load on every link.
        let mut w_prev = [0.0; 3];
        let mut a_prev = [0.0; 3];
        let mut v_prev = [-gravity[0], -gravity[1], -gravity[2]];

        let mut i = 0;
        while i < N {
            let spec = joints[i];
            // Also validates the coordinate against the joint type and limits.
            let step = spec
                .origin_transform()
                .compose(spec.motion_transform(input.positions[i])?);
            let m = step.to_matrix();
            let r = [
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ];
            let p = [m[0][3], m[1][3], m[2][3]];
            rot[i] = r;
            org[i] = p;
            let axis = spec.axis.unit_vector();
            let rtw = mat_t_vec(&r, &w_prev);
            let rta = mat_t_vec(&r, &a_prev);
            let transported = mat_t_vec(
                &r,
                &add3(
                    v_prev,
                    add3(cross3(a_prev, p), cross3(w_prev, cross3(w_prev, p))),
                ),
            );
            match spec.joint_type {
                CuJointType::Revolute => {
                    omega[i] = add3(rtw, scale3(axis, qd[i]));
                    alpha[i] = add3(
                        add3(rta, scale3(cross3(rtw, axis), qd[i])),
                        scale3(axis, qdd[i]),
                    );
                    acc[i] = transported;
                }
                CuJointType::Prismatic => {
                    omega[i] = rtw;
                    alpha[i] = rta;
                    acc[i] = add3(
                        add3(transported, scale3(cross3(rtw, axis), 2.0 * qd[i])),
                        scale3(axis, qdd[i]),
                    );
                }
                CuJointType::Fixed => {
                    omega[i] = rtw;
                    alpha[i] = rta;
                    acc[i] = transported;
                }
            }
            let com = links[i].center_of_mass.map(|c| c.get::<meter>());
            acc_com[i] = add3(
                add3(acc[i], cross3(alpha[i], com)),
                cross3(omega[i], cross3(omega[i], com)),
            );
            w_prev = omega[i];
            a_prev = alpha[i];
            v_prev = acc[i];
            i += 1;
        }

        let mut f_next = [0.0; 3];
        let mut mu_next = [0.0; 3];
        let mut i = N;
        while i > 0 {
            i -= 1;
            let spec = joints[i];
            let link = links[i];
            let mass = link.mass.get::<kilogram>();
            let com = link.center_of_mass.map(|c| c.get::<meter>());
            let tensor = link.inertia.map(|v| v.get::<kilogram_square_meter>());
            let (rf, rmu, pp) = if i + 1 < N {
                (
                    mat_vec(&rot[i + 1], &f_next),
                    mat_vec(&rot[i + 1], &mu_next),
                    org[i + 1],
                )
            } else {
                (f_next, mu_next, [0.0; 3])
            };
            let f = add3(rf, scale3(acc_com[i], mass));
            let gyro = add3(
                inertia_mat_vec(&tensor, &alpha[i]),
                cross3(omega[i], inertia_mat_vec(&tensor, &omega[i])),
            );
            let mu = add3(
                add3(
                    add3(rmu, cross3(pp, rf)),
                    cross3(com, scale3(acc_com[i], mass)),
                ),
                gyro,
            );
            let axis = spec.axis.unit_vector();
            out_efforts[i] = match spec.joint_type {
                CuJointType::Revolute => {
                    CuJointEffort::Revolute(Torque::new::<newton_meter>(dot3(axis, mu)))
                }
                CuJointType::Prismatic => {
                    CuJointEffort::Prismatic(Force::new::<newton>(dot3(axis, f)))
                }
                CuJointType::Fixed => CuJointEffort::Fixed,
            };
            f_next = f;
            mu_next = mu;
        }
        Ok(())
    }

    /// Computes the joint efforts that hold the chain still at `positions`, in place.
    ///
    /// This is [`Self::inverse_dynamics`] with zero velocity and acceleration,
    /// so the result is the gravity load alone.
    ///
    /// # Example
    /// ```rust
    /// use cu29::units::si::acceleration::meter_per_second_squared;
    /// use cu29::units::si::angle::radian;
    /// use cu29::units::si::angular_velocity::radian_per_second;
    /// use cu29::units::si::f64::Acceleration;
    /// use cu29::units::si::f64::Angle;
    /// use cu29::units::si::f64::AngularVelocity;
    /// use cu29::units::si::f64::Length;
    /// use cu29::units::si::f64::Mass;
    /// use cu29::units::si::f64::MomentOfInertia;
    /// use cu29::units::si::length::meter;
    /// use cu29::units::si::mass::kilogram;
    /// use cu29::units::si::moment_of_inertia::kilogram_square_meter;
    /// use repose_kinematics::CuChainInertia;
    /// use repose_kinematics::CuJointAxis;
    /// use repose_kinematics::CuJointEffort;
    /// use repose_kinematics::CuJointLimits;
    /// use repose_kinematics::CuJointSpec;
    /// use repose_kinematics::CuKinematicChain;
    /// use repose_kinematics::CuLinkInertia;
    ///
    /// let chain = CuKinematicChain::new([CuJointSpec::revolute(
    ///     [Length::new::<meter>(0.0); 3],
    ///     [Angle::new::<radian>(0.0); 3],
    ///     CuJointAxis::PosY,
    ///     CuJointLimits::revolute(
    ///         Angle::new::<radian>(-core::f64::consts::PI),
    ///         Angle::new::<radian>(core::f64::consts::PI),
    ///         AngularVelocity::new::<radian_per_second>(2.0),
    ///     ),
    /// )]);
    /// let inertia = CuChainInertia::new([CuLinkInertia::new(
    ///     Mass::new::<kilogram>(2.0),
    ///     [
    ///         Length::new::<meter>(0.25),
    ///         Length::new::<meter>(0.0),
    ///         Length::new::<meter>(0.0),
    ///     ],
    ///     [MomentOfInertia::new::<kilogram_square_meter>(0.0); 6],
    /// )
    /// .unwrap()])
    /// .unwrap();
    /// let q = [Angle::new::<radian>(0.0).into()];
    /// let gravity = [
    ///     Acceleration::new::<meter_per_second_squared>(0.0),
    ///     Acceleration::new::<meter_per_second_squared>(0.0),
    ///     Acceleration::new::<meter_per_second_squared>(-9.81),
    /// ];
    /// let mut efforts = [CuJointEffort::default(); 1];
    /// chain
    ///     .gravity_compensation(&inertia, &q, gravity, &mut efforts)
    ///     .unwrap();
    /// // Holding torque: -m g l = -2 * 9.81 * 0.25 N·m.
    /// assert!((efforts[0].si_value() + 4.905).abs() < 1e-9);
    /// ```
    pub fn gravity_compensation(
        &self,
        inertia: &CuChainInertia<N>,
        positions: &[CuJointPosition; N],
        gravity: [Acceleration; 3],
        out_efforts: &mut [CuJointEffort; N],
    ) -> CuResult<()> {
        let mut velocities = [CuJointVelocity::default(); N];
        let mut accelerations = [CuJointAcceleration::default(); N];
        let mut i = 0;
        while i < N {
            velocities[i] = CuJointVelocity::from_coordinate(positions[i], 0.0);
            accelerations[i] = CuJointAcceleration::from_coordinate(positions[i], 0.0);
            i += 1;
        }
        let input = CuInverseDynamicsInput::new(positions, &velocities, &accelerations, gravity);
        self.inverse_dynamics(inertia, &input, out_efforts)
    }
}

fn velocity_matches(position: CuJointPosition, velocity: CuJointVelocity) -> bool {
    matches!(
        (position, velocity),
        (CuJointPosition::Revolute(_), CuJointVelocity::Revolute(_))
            | (CuJointPosition::Prismatic(_), CuJointVelocity::Prismatic(_))
            | (CuJointPosition::Fixed, CuJointVelocity::Fixed)
    )
}

fn acceleration_matches(position: CuJointPosition, acceleration: CuJointAcceleration) -> bool {
    matches!(
        (position, acceleration),
        (
            CuJointPosition::Revolute(_),
            CuJointAcceleration::Revolute(_)
        ) | (
            CuJointPosition::Prismatic(_),
            CuJointAcceleration::Prismatic(_)
        ) | (CuJointPosition::Fixed, CuJointAcceleration::Fixed)
    )
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mat_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [dot3(m[0], *v), dot3(m[1], *v), dot3(m[2], *v)]
}

fn mat_t_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiplies the symmetric tensor `[xx, yy, zz, xy, xz, yz]` by a vector.
fn inertia_mat_vec(tensor: &[f64; 6], v: &[f64; 3]) -> [f64; 3] {
    [
        tensor[0] * v[0] + tensor[3] * v[1] + tensor[4] * v[2],
        tensor[3] * v[0] + tensor[1] * v[1] + tensor[5] * v[2],
        tensor[4] * v[0] + tensor[5] * v[1] + tensor[2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::CuChainInertia;
    use super::CuInverseDynamicsInput;
    use super::CuLinkInertia;
    use super::standard_gravity;
    use crate::CuJointAcceleration;
    use crate::CuJointAxis;
    use crate::CuJointEffort;
    use crate::CuJointLimits;
    use crate::CuJointPosition;
    use crate::CuJointSpec;
    use crate::CuJointVelocity;
    use crate::CuKinematicChain;
    use crate::igus_rebel_chain;
    use cu_spatial_payloads::Transform3D;
    use cu29::units::si::acceleration::meter_per_second_squared;
    use cu29::units::si::angle::radian;
    use cu29::units::si::angular_acceleration::radian_per_second_squared;
    use cu29::units::si::angular_velocity::radian_per_second;
    use cu29::units::si::f64::Acceleration;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::AngularAcceleration;
    use cu29::units::si::f64::AngularVelocity;
    use cu29::units::si::f64::Length;
    use cu29::units::si::f64::Mass;
    use cu29::units::si::f64::MomentOfInertia;
    use cu29::units::si::f64::Velocity;
    use cu29::units::si::length::meter;
    use cu29::units::si::mass::kilogram;
    use cu29::units::si::moment_of_inertia::kilogram_square_meter;
    use cu29::units::si::velocity::meter_per_second;

    const G: f64 = 9.81;

    fn kg(v: f64) -> Mass {
        Mass::new::<kilogram>(v)
    }
    fn m3(v: [f64; 3]) -> [Length; 3] {
        v.map(Length::new::<meter>)
    }
    fn i6(v: [f64; 6]) -> [MomentOfInertia; 6] {
        v.map(MomentOfInertia::new::<kilogram_square_meter>)
    }
    fn gravity_down() -> [Acceleration; 3] {
        [0.0, 0.0, -G].map(Acceleration::new::<meter_per_second_squared>)
    }
    fn rev_limits() -> CuJointLimits {
        CuJointLimits::revolute(
            Angle::new::<radian>(-core::f64::consts::PI),
            Angle::new::<radian>(core::f64::consts::PI),
            AngularVelocity::new::<radian_per_second>(10.0),
        )
    }
    fn rev_pos(q: f64) -> CuJointPosition {
        Angle::new::<radian>(q).into()
    }
    fn rev_vel(qd: f64) -> CuJointVelocity {
        CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(qd))
    }
    fn rev_acc(qdd: f64) -> CuJointAcceleration {
        CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(qdd))
    }
    fn link(mass: f64, com: [f64; 3], tensor: [f64; 6]) -> CuLinkInertia {
        CuLinkInertia::new(kg(mass), m3(com), i6(tensor)).unwrap()
    }

    #[test]
    fn pendulum_matches_closed_form() {
        let chain = CuKinematicChain::new([CuJointSpec::revolute(
            m3([0.0; 3]),
            [Angle::new::<radian>(0.0); 3],
            CuJointAxis::PosY,
            rev_limits(),
        )]);
        let mass = 2.0;
        let lc = 0.25;
        let iyy = 0.02;
        let inertia = CuChainInertia::new([link(
            mass,
            [lc, 0.0, 0.0],
            [0.01, iyy, 0.015, 0.0, 0.0, 0.0],
        )])
        .unwrap();
        // Pivot inertia for the closed form τ = (I + m l²) q̈ − m g l cos q.
        let pivot = iyy + mass * lc * lc;
        for (q, qd, qdd) in [
            (0.0, 0.0, 0.0),
            (0.3, 1.5, -2.0),
            (-1.2, -0.7, 0.9),
            (2.5, 3.0, -4.0),
        ] {
            let positions = [rev_pos(q)];
            let velocities = [rev_vel(qd)];
            let accelerations = [rev_acc(qdd)];
            let input = CuInverseDynamicsInput::new(
                &positions,
                &velocities,
                &accelerations,
                gravity_down(),
            );
            let mut efforts = [CuJointEffort::default(); 1];
            chain
                .inverse_dynamics(&inertia, &input, &mut efforts)
                .unwrap();
            let expected = pivot * qdd - mass * G * lc * q.cos();
            assert!(
                (efforts[0].si_value() - expected).abs() < 1e-9,
                "q={q} qd={qd} qdd={qdd}: got {}, want {expected}",
                efforts[0].si_value()
            );
            let mut held = [CuJointEffort::default(); 1];
            chain
                .gravity_compensation(&inertia, &positions, gravity_down(), &mut held)
                .unwrap();
            assert!((held[0].si_value() + mass * G * lc * q.cos()).abs() < 1e-9);
        }
    }

    #[test]
    fn planar_two_link_matches_lagrangian() {
        let l1 = 0.5;
        let lc1 = 0.25;
        let lc2 = 0.2;
        let m1 = 2.0;
        let m2 = 1.5;
        let i1 = 0.02;
        let i2 = 0.01;
        let chain = CuKinematicChain::new([
            CuJointSpec::revolute(
                m3([0.0; 3]),
                [Angle::new::<radian>(0.0); 3],
                CuJointAxis::PosY,
                rev_limits(),
            ),
            CuJointSpec::revolute(
                m3([l1, 0.0, 0.0]),
                [Angle::new::<radian>(0.0); 3],
                CuJointAxis::PosY,
                rev_limits(),
            ),
        ]);
        let inertia = CuChainInertia::new([
            link(m1, [lc1, 0.0, 0.0], [0.01, i1, 0.015, 1e-4, 0.0, 0.0]),
            link(m2, [lc2, 0.0, 0.0], [0.008, i2, 0.009, 0.0, 1e-4, 0.0]),
        ])
        .unwrap();
        for (q1, q2, qd1, qd2, qdd1, qdd2) in [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.4, -0.7, 1.2, -0.5, 2.0, 1.0),
            (-1.0, 2.0, -2.5, 1.5, -1.0, 3.0),
            (0.1, 0.2, 0.0, 0.0, 5.0, -5.0),
        ] {
            let positions = [rev_pos(q1), rev_pos(q2)];
            let velocities = [rev_vel(qd1), rev_vel(qd2)];
            let accelerations = [rev_acc(qdd1), rev_acc(qdd2)];
            let input = CuInverseDynamicsInput::new(
                &positions,
                &velocities,
                &accelerations,
                gravity_down(),
            );
            let mut efforts = [CuJointEffort::default(); 2];
            chain
                .inverse_dynamics(&inertia, &input, &mut efforts)
                .unwrap();
            // Hand-derived M(q) q̈ + C(q, q̇) + G(q) for links along +X at zero,
            // positive rotation about +Y, gravity −Z.
            let s2 = q2.sin();
            let c2 = q2.cos();
            let m11 = m1 * lc1 * lc1 + i1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * c2) + i2;
            let m12 = m2 * (lc2 * lc2 + l1 * lc2 * c2) + i2;
            let m22 = m2 * lc2 * lc2 + i2;
            let c1 = -m2 * l1 * lc2 * s2 * (2.0 * qd1 * qd2 + qd2 * qd2);
            let c2 = m2 * l1 * lc2 * s2 * qd1 * qd1;
            let g1 = -G * (m1 * lc1 * q1.cos() + m2 * l1 * q1.cos() + m2 * lc2 * (q1 + q2).cos());
            let g2 = -G * m2 * lc2 * (q1 + q2).cos();
            let expected = [
                m11 * qdd1 + m12 * qdd2 + c1 + g1,
                m12 * qdd1 + m22 * qdd2 + c2 + g2,
            ];
            for (j, (effort, want)) in efforts.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (effort.si_value() - want).abs() < 1e-9,
                    "joint {j}: got {}, want {want}",
                    effort.si_value()
                );
            }
        }
    }

    #[test]
    fn prismatic_matches_closed_form() {
        let zeros = [Angle::new::<radian>(0.0); 3];
        let slider = |axis| {
            CuKinematicChain::new([CuJointSpec::prismatic(
                m3([0.0; 3]),
                zeros,
                axis,
                CuJointLimits::prismatic(
                    Length::new::<meter>(0.0),
                    Length::new::<meter>(1.0),
                    Velocity::new::<meter_per_second>(1.0),
                ),
            )])
        };
        let mass = 3.0;
        // Off-centre mass exercises the transport terms without changing the
        // closed form: at rest the force is inertia plus the gravity projection.
        let inertia = CuChainInertia::new([link(
            mass,
            [0.05, -0.02, 0.1],
            [0.01, 0.012, 0.008, 1e-4, 0.0, -1e-4],
        )])
        .unwrap();
        for (chain, projection) in [
            (slider(CuJointAxis::PosZ), G),
            (slider(CuJointAxis::PosX), 0.0),
        ] {
            for (d, vd, ad) in [(0.0, 0.0, 0.0), (0.2, 0.5, -1.0), (0.35, -1.2, 2.5)] {
                let positions = [CuJointPosition::Prismatic(Length::new::<meter>(d))];
                let velocities = [CuJointVelocity::Prismatic(
                    Velocity::new::<meter_per_second>(vd),
                )];
                let accelerations = [CuJointAcceleration::from_coordinate(positions[0], ad)];
                let input = CuInverseDynamicsInput::new(
                    &positions,
                    &velocities,
                    &accelerations,
                    gravity_down(),
                );
                let mut efforts = [CuJointEffort::default(); 1];
                chain
                    .inverse_dynamics(&inertia, &input, &mut efforts)
                    .unwrap();
                assert!(matches!(efforts[0], CuJointEffort::Prismatic(_)));
                let expected = mass * (ad + projection);
                assert!(
                    (efforts[0].si_value() - expected).abs() < 1e-9,
                    "got {}, want {expected}",
                    efforts[0].si_value()
                );
            }
        }
    }

    #[test]
    fn mixed_chain_reports_effort_kinds_and_rejects_mismatches() {
        let zeros = [Angle::new::<radian>(0.0); 3];
        let chain = CuKinematicChain::new([
            CuJointSpec::revolute(m3([0.0, 0.0, 0.1]), zeros, CuJointAxis::PosZ, rev_limits()),
            CuJointSpec::prismatic(
                m3([0.0; 3]),
                zeros,
                CuJointAxis::PosX,
                CuJointLimits::prismatic(
                    Length::new::<meter>(0.0),
                    Length::new::<meter>(1.0),
                    Velocity::new::<meter_per_second>(1.0),
                ),
            ),
            CuJointSpec {
                joint_type: crate::CuJointType::Fixed,
                ..Default::default()
            },
        ]);
        let inertia = CuChainInertia::new([
            link(1.0, [0.05, 0.0, 0.0], [0.005, 0.006, 0.004, 0.0, 0.0, 0.0]),
            link(0.5, [0.0, 0.0, 0.02], [0.002, 0.002, 0.001, 0.0, 0.0, 0.0]),
            CuLinkInertia::default(),
        ])
        .unwrap();
        let positions = [
            rev_pos(0.3),
            CuJointPosition::Prismatic(Length::new::<meter>(0.1)),
            CuJointPosition::Fixed,
        ];
        let velocities = [
            rev_vel(0.5),
            CuJointVelocity::Prismatic(Velocity::new::<meter_per_second>(0.1)),
            CuJointVelocity::Fixed,
        ];
        let accelerations = [
            rev_acc(-0.2),
            CuJointAcceleration::from_coordinate(positions[1], 0.3),
            CuJointAcceleration::Fixed,
        ];
        let input =
            CuInverseDynamicsInput::new(&positions, &velocities, &accelerations, gravity_down());
        let mut efforts = [CuJointEffort::default(); 3];
        chain
            .inverse_dynamics(&inertia, &input, &mut efforts)
            .unwrap();
        assert!(matches!(efforts[0], CuJointEffort::Revolute(_)));
        assert!(matches!(efforts[1], CuJointEffort::Prismatic(_)));
        assert!(matches!(efforts[2], CuJointEffort::Fixed));
        assert_eq!(efforts[2].si_value(), 0.0);

        let mut bad_velocities = velocities;
        bad_velocities[1] = rev_vel(0.1);
        let bad = CuInverseDynamicsInput::new(
            &positions,
            &bad_velocities,
            &accelerations,
            gravity_down(),
        );
        assert!(
            chain
                .inverse_dynamics(&inertia, &bad, &mut efforts)
                .is_err()
        );
        let mut bad_accelerations = accelerations;
        bad_accelerations[0] = CuJointAcceleration::from_coordinate(positions[1], 0.1);
        let bad = CuInverseDynamicsInput::new(
            &positions,
            &velocities,
            &bad_accelerations,
            gravity_down(),
        );
        assert!(
            chain
                .inverse_dynamics(&inertia, &bad, &mut efforts)
                .is_err()
        );
        let bad_gravity = [f64::NAN, 0.0, -G].map(Acceleration::new::<meter_per_second_squared>);
        let bad = CuInverseDynamicsInput::new(&positions, &velocities, &accelerations, bad_gravity);
        assert!(
            chain
                .inverse_dynamics(&inertia, &bad, &mut efforts)
                .is_err()
        );
        let mut bad_velocities = velocities;
        bad_velocities[0] = rev_vel(f64::INFINITY);
        let bad = CuInverseDynamicsInput::new(
            &positions,
            &bad_velocities,
            &accelerations,
            gravity_down(),
        );
        assert!(
            chain
                .inverse_dynamics(&inertia, &bad, &mut efforts)
                .is_err()
        );
    }

    /// Plausible made-up inertias for the 6-DOF igus ReBeL preset.
    fn rebel_inertia() -> CuChainInertia<6> {
        CuChainInertia::new([
            link(
                2.2,
                [0.0, 0.01, 0.05],
                [0.02, 0.022, 0.012, 1e-4, -2e-4, 1e-4],
            ),
            link(
                1.6,
                [0.0, -0.02, 0.04],
                [0.015, 0.012, 0.018, 2e-4, 1e-4, -1e-4],
            ),
            link(
                1.1,
                [0.0, 0.015, -0.03],
                [0.008, 0.009, 0.006, -1e-4, 1e-4, 2e-4],
            ),
            link(
                0.7,
                [0.01, 0.0, 0.02],
                [0.004, 0.0045, 0.003, 5e-5, -5e-5, 5e-5],
            ),
            link(
                0.4,
                [0.0, 0.005, 0.01],
                [0.0015, 0.0012, 0.0018, 2e-5, 1e-5, -2e-5],
            ),
            link(
                0.25,
                [0.0, 0.0, 0.03],
                [8e-4, 7e-4, 9e-4, 1e-5, -1e-5, 1e-5],
            ),
        ])
        .unwrap()
    }

    /// Gravitational potential energy `V = −Σ m (g·p)` from link-frame centres of mass.
    fn potential(
        chain: &CuKinematicChain<6>,
        inertia: &CuChainInertia<6>,
        q: &[CuJointPosition; 6],
        gravity: [f64; 3],
    ) -> f64 {
        let mut frames = [Transform3D::<f64>::identity(); 6];
        let mut ee = Transform3D::<f64>::identity();
        chain.forward_frames(q, &mut frames, &mut ee).unwrap();
        let mut v = 0.0;
        for (frame, link) in frames.iter().zip(inertia.links()) {
            let m = frame.to_matrix();
            let c = link.center_of_mass.map(|c| c.get::<meter>());
            let p = [
                m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2] + m[0][3],
                m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2] + m[1][3],
                m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2] + m[2][3],
            ];
            v -= link.mass.get::<kilogram>()
                * (gravity[0] * p[0] + gravity[1] * p[1] + gravity[2] * p[2]);
        }
        v
    }

    #[test]
    fn rebel_gravity_matches_potential_gradient() {
        let chain = igus_rebel_chain();
        let inertia = rebel_inertia();
        let gravity = standard_gravity();
        let g = gravity.map(|a| a.get::<meter_per_second_squared>());
        for q in [
            [0.0; 6],
            [0.25, -0.4, 0.55, -0.3, 0.45, 0.15],
            [-0.5, 0.9, -1.1, 0.7, -0.8, 1.2],
        ] {
            let positions = q.map(rev_pos);
            let mut efforts = [CuJointEffort::default(); 6];
            chain
                .gravity_compensation(&inertia, &positions, gravity, &mut efforts)
                .unwrap();
            let eps = 1e-7;
            let mut j = 0;
            while j < 6 {
                let mut plus = q;
                let mut minus = q;
                plus[j] += eps;
                minus[j] -= eps;
                let grad = (potential(&chain, &inertia, &plus.map(rev_pos), g)
                    - potential(&chain, &inertia, &minus.map(rev_pos), g))
                    / (2.0 * eps);
                assert!(
                    (efforts[j].si_value() - grad).abs() < 1e-6,
                    "joint {j}: gravity torque {}, potential gradient {grad}",
                    efforts[j].si_value()
                );
                j += 1;
            }
        }
    }

    /// Joint-space mass matrix of the ReBeL at `q`: column `j` is the effort
    /// for a unit acceleration of joint `j` with no velocity and no gravity.
    fn rebel_mass_matrix(q: [f64; 6]) -> [[f64; 6]; 6] {
        let chain = igus_rebel_chain();
        let inertia = rebel_inertia();
        let positions = q.map(rev_pos);
        let velocities = [0.0; 6].map(rev_vel);
        let zero_g = [0.0; 3].map(Acceleration::new::<meter_per_second_squared>);
        let mut matrix = [[0.0; 6]; 6];
        for j in 0..6 {
            let mut qdd = [0.0; 6];
            qdd[j] = 1.0;
            let accelerations = qdd.map(rev_acc);
            let input =
                CuInverseDynamicsInput::new(&positions, &velocities, &accelerations, zero_g);
            let mut efforts = [CuJointEffort::default(); 6];
            chain
                .inverse_dynamics(&inertia, &input, &mut efforts)
                .unwrap();
            for (row, effort) in matrix.iter_mut().zip(efforts) {
                row[j] = effort.si_value();
            }
        }
        matrix
    }

    #[test]
    fn rebel_mass_matrix_is_symmetric_positive_definite() {
        let m = rebel_mass_matrix([0.25, -0.4, 0.55, -0.3, 0.45, 0.15]);
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!(
                    (value - m[j][i]).abs() < 1e-12,
                    "M[{i}][{j}] = {value}, M[{j}][{i}] = {}",
                    m[j][i]
                );
            }
        }
        // Kinetic energy is positive for every nonzero joint velocity tried.
        for v in [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.7, 0.2, 1.1, -0.4, 0.9],
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ] {
            let mut energy = 0.0;
            for i in 0..6 {
                for j in 0..6 {
                    energy += 0.5 * v[i] * m[i][j] * v[j];
                }
            }
            assert!(energy > 0.0, "kinetic energy {energy} for velocity {v:?}");
        }
    }

    /// Power balance in 3D: with no gravity and no acceleration, the power
    /// `q̇·τ` of the velocity-dependent efforts equals the rate of change of
    /// kinetic energy, `½ q̇ᵀ Ṁ q̇`. This checks the Coriolis, centrifugal and
    /// gyroscopic terms on a spatial arm with full inertia tensors.
    #[test]
    fn rebel_velocity_terms_conserve_energy() {
        let chain = igus_rebel_chain();
        let inertia = rebel_inertia();
        let zero_g = [0.0; 3].map(Acceleration::new::<meter_per_second_squared>);
        for (q, qd) in [
            (
                [0.25, -0.4, 0.55, -0.3, 0.45, 0.15],
                [0.6, -0.9, 0.4, 1.2, -0.7, 0.5],
            ),
            (
                [-0.5, 0.9, -1.1, 0.7, -0.8, 1.2],
                [-1.1, 0.3, 0.8, -0.6, 1.0, -0.4],
            ),
        ] {
            let positions = q.map(rev_pos);
            let velocities = qd.map(rev_vel);
            let accelerations = [0.0; 6].map(rev_acc);
            let input =
                CuInverseDynamicsInput::new(&positions, &velocities, &accelerations, zero_g);
            let mut efforts = [CuJointEffort::default(); 6];
            chain
                .inverse_dynamics(&inertia, &input, &mut efforts)
                .unwrap();
            let mut power = 0.0;
            for i in 0..6 {
                power += qd[i] * efforts[i].si_value();
            }

            let h = 1e-5;
            let mut ahead = q;
            let mut behind = q;
            for i in 0..6 {
                ahead[i] += h * qd[i];
                behind[i] -= h * qd[i];
            }
            let m_ahead = rebel_mass_matrix(ahead);
            let m_behind = rebel_mass_matrix(behind);
            let mut energy_rate = 0.0;
            for i in 0..6 {
                for j in 0..6 {
                    energy_rate +=
                        0.5 * qd[i] * (m_ahead[i][j] - m_behind[i][j]) / (2.0 * h) * qd[j];
                }
            }
            assert!(
                power.abs() > 1e-4,
                "test state has no velocity-dependent power to compare"
            );
            assert!(
                (power - energy_rate).abs() < 1e-6,
                "power {power}, kinetic energy rate {energy_rate}"
            );
        }
    }

    #[test]
    fn zero_motion_zero_gravity_gives_zero_torque() {
        let chain = igus_rebel_chain();
        let inertia = rebel_inertia();
        let positions = [0.2, -0.5, 0.8, -0.1, 0.6, -0.9].map(rev_pos);
        let velocities = positions.map(|p| CuJointVelocity::from_coordinate(p, 0.0));
        let accelerations = positions.map(|p| CuJointAcceleration::from_coordinate(p, 0.0));
        let zero_g = [0.0; 3].map(Acceleration::new::<meter_per_second_squared>);
        let input = CuInverseDynamicsInput::new(&positions, &velocities, &accelerations, zero_g);
        let mut efforts = [CuJointEffort::default(); 6];
        chain
            .inverse_dynamics(&inertia, &input, &mut efforts)
            .unwrap();
        for effort in efforts {
            assert!(effort.si_value().abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_inertia_rejected() {
        let com = m3([0.1, 0.0, 0.0]);
        let good = i6([0.02, 0.022, 0.012, 1e-4, -2e-4, 1e-4]);
        assert!(CuLinkInertia::new(kg(-1.0), com, good).is_err());
        assert!(CuLinkInertia::new(kg(f64::NAN), com, good).is_err());
        assert!(CuLinkInertia::new(kg(1.0), m3([f64::INFINITY, 0.0, 0.0]), good).is_err());
        let mut bad = good;
        bad[0] = MomentOfInertia::new::<kilogram_square_meter>(f64::NAN);
        assert!(CuLinkInertia::new(kg(1.0), com, bad).is_err());
        let mut bad = good;
        bad[2] = MomentOfInertia::new::<kilogram_square_meter>(-0.5);
        assert!(CuLinkInertia::new(kg(1.0), com, bad).is_err());
        // Triangle violation: 0.01 + 0.01 < 0.05.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.01, 0.01, 0.05, 0.0, 0.0, 0.0])).is_err());
        // Minor violation: xy² > xx·yy.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.01, 0.01, 0.01, 0.02, 0.0, 0.0])).is_err());
        // Determinant violation despite valid minors: det < 0.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([1.0, 1.0, 1.0, 0.9, 0.9, -0.9])).is_err());
        // Zero mass with nonzero inertia.
        assert!(CuLinkInertia::new(kg(0.0), com, good).is_err());
        // Valid: full tensor, point mass, and massless link.
        assert!(CuLinkInertia::new(kg(1.0), com, good).is_ok());
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.0; 6])).is_ok());
        assert!(CuLinkInertia::new(kg(0.0), m3([0.0; 3]), i6([0.0; 6])).is_ok());
        assert!(CuChainInertia::new([CuLinkInertia::new(kg(1.0), com, good).unwrap()]).is_ok());
    }

    #[test]
    fn principal_moment_triangle_violation_rejected() {
        let com = m3([0.0; 3]);
        // Diagonal triangle checks and positive semi-definiteness both hold,
        // but the principal moments are 2.8, 0.1, 0.1 and 2.8 > 0.1 + 0.1.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([1.0, 1.0, 1.0, 0.9, 0.9, 0.9])).is_err());
        // Principal moments 1.5, 0.5, 0.5 with an unremarkable diagonal.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])).is_err());
        // A point mass has no room for products of inertia.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0])).is_err());
    }

    #[test]
    fn valid_inertia_edge_cases_accepted() {
        let com = m3([0.0, 0.01, -0.02]);
        // Point mass.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.0; 6])).is_ok());
        // Thin rod along x: one zero principal moment, the other two equal.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.0, 0.1, 0.1, 0.0, 0.0, 0.0])).is_ok());
        // Flat plate in the xy plane: exactly on the boundary, zz = xx + yy.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])).is_ok());
        // The same plate turned 45° about z: principal moments 0.1, 0.2, 0.3
        // still sit on the boundary, now with a product of inertia.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.15, 0.15, 0.3, 0.05, 0.0, 0.0])).is_ok());
        // Thin rod along the (1, 1, 1) diagonal: principal moments 0, 0.3, 0.3
        // with every product of inertia nonzero.
        assert!(CuLinkInertia::new(kg(1.0), com, i6([0.2, 0.2, 0.2, -0.1, -0.1, -0.1])).is_ok());
        // Boundary tensor carrying the rounding of a tabulated value.
        assert!(
            CuLinkInertia::new(kg(1.0), com, i6([0.1, 0.2, 0.3 + 1e-12, 0.0, 0.0, 0.0])).is_ok()
        );
        // Realistic slender arm link with all three products of inertia.
        assert!(
            CuLinkInertia::new(
                kg(4.97),
                com,
                i6([0.7034, 0.7066, 0.0091, -1.39e-4, 6.772e-3, 1.9169e-2])
            )
            .is_ok()
        );
    }

    fn pendulum_chain() -> CuKinematicChain<1> {
        CuKinematicChain::new([CuJointSpec::revolute(
            m3([0.0; 3]),
            [Angle::new::<radian>(0.0); 3],
            CuJointAxis::PosY,
            rev_limits(),
        )])
    }

    #[test]
    fn deserialized_chain_inertia_is_validated() {
        let good =
            "(links:((mass:2.0,center_of_mass:(0.25,0.0,0.0),inertia:(0.0,0.0,0.0,0.0,0.0,0.0))))";
        let inertia: CuChainInertia<1> = ron::from_str(good).unwrap();
        assert_eq!(
            inertia,
            CuChainInertia::new([link(2.0, [0.25, 0.0, 0.0], [0.0; 6])]).unwrap()
        );
        assert_eq!(ron::to_string(&inertia).unwrap(), good);

        // Negative mass must not come back as a usable chain inertia: it would
        // flip the sign of every gravity compensation effort.
        let negative_mass =
            "(links:((mass:-2.0,center_of_mass:(0.25,0.0,0.0),inertia:(0.0,0.0,0.0,0.0,0.0,0.0))))";
        assert!(ron::from_str::<CuChainInertia<1>>(negative_mass).is_err());
        let bad_tensor =
            "(links:((mass:2.0,center_of_mass:(0.25,0.0,0.0),inertia:(1.0,1.0,1.0,0.9,0.9,0.9))))";
        assert!(ron::from_str::<CuChainInertia<1>>(bad_tensor).is_err());
    }

    #[test]
    fn inverse_dynamics_rejects_unvalidated_inertia() {
        // `CuLinkInertia` has public fields, so a struct literal skips its
        // constructor. Building the chain inertia in-module stands in for any
        // path that bypasses `CuChainInertia::new`, such as reflection.
        let inertia = CuChainInertia {
            links: [CuLinkInertia {
                mass: kg(-2.0),
                center_of_mass: m3([0.25, 0.0, 0.0]),
                inertia: i6([0.0; 6]),
            }],
        };
        let chain = pendulum_chain();
        let positions = [rev_pos(0.0)];
        let mut efforts = [CuJointEffort::default(); 1];
        assert!(
            chain
                .gravity_compensation(&inertia, &positions, gravity_down(), &mut efforts)
                .is_err()
        );
        let velocities = [rev_vel(0.0)];
        let accelerations = [rev_acc(0.0)];
        let input =
            CuInverseDynamicsInput::new(&positions, &velocities, &accelerations, gravity_down());
        assert!(
            chain
                .inverse_dynamics(&inertia, &input, &mut efforts)
                .is_err()
        );
    }
}
