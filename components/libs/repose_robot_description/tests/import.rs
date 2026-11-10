#![cfg(feature = "std")]

use cu_spatial_payloads::Transform3D;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Mass;
use cu29::units::si::f64::MomentOfInertia;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::mass::kilogram;
use cu29::units::si::moment_of_inertia::kilogram_square_meter;
use cu29::units::si::velocity::meter_per_second;
use repose_collision::CuCollisionEnv;
use repose_collision::CuCollisionModelParams;
use repose_collision::CuCollisionReport;
use repose_collision::CuCollisionShape;
use repose_collision::CuSphere;
use repose_kinematics::CuDiffIkParams;
use repose_kinematics::CuIkParams;
use repose_kinematics::CuIkStatus;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuTwist6D;
use repose_kinematics::CuTwistFrame;
use repose_robot_description::CuRobotDescription;
use repose_robot_description::CuSrdfImport;
use repose_robot_description::CuUrdfImport;

const URDF: &str = include_str!("fixtures/serial.urdf");
const SRDF: &str = include_str!("fixtures/serial.srdf");
fn collision() -> CuCollisionModelParams {
    CuCollisionModelParams {
        safety_margin: Length::new::<meter>(0.005),
        skip_adjacent_links: false,
    }
}
fn urdf_import<'a>(base_link: &'a str, tip_link: &'a str) -> CuUrdfImport<'a> {
    CuUrdfImport::new(base_link, tip_link).with_collision(collision())
}
fn robot() -> CuRobotDescription<2, 6> {
    CuRobotDescription::from_urdf_srdf(
        URDF,
        SRDF,
        CuSrdfImport::new("manipulator").with_collision(collision()),
    )
    .unwrap()
}
fn joints(angle: f64, distance: f64) -> [CuJointPosition; 2] {
    [
        Angle::new::<radian>(angle).into(),
        Length::new::<meter>(distance).into(),
    ]
}

#[test]
fn fixed_frames_and_prismatic_units_match_analytic_fk_and_jacobian() {
    let robot = robot();
    assert_eq!(robot.joint_names(), &["swing", "extension"]);
    assert_eq!(robot.collision_model().shape_count(), 6);
    let q = joints(0.0, 0.2);
    let mut pose = Transform3D::<f64>::identity();
    robot.chain().forward_kinematics(&q, &mut pose).unwrap();
    let m = pose.to_matrix();
    assert!(m[0][3].abs() < 1e-12);
    assert!((m[1][3] - 0.9).abs() < 1e-12);
    assert!((m[2][3] - 0.2).abs() < 1e-12);
    let mut jac = [[0.0; 2]; 6];
    robot.chain().geometric_jacobian(&q, &mut jac).unwrap();
    for col in 0..2 {
        let mut perturbed = q;
        perturbed[col] = q[col].with_si_value(q[col].si_value() + 1e-7);
        let mut p = pose;
        robot
            .chain()
            .forward_kinematics(&perturbed, &mut p)
            .unwrap();
        for row in 0..3 {
            assert!(((p.to_matrix()[row][3] - m[row][3]) / 1e-7 - jac[row][col]).abs() < 1e-6);
        }
    }
    assert_eq!(jac[5][1], 0.0);
    let wrong_units = [q[0], Angle::new::<radian>(0.2).into()];
    assert!(
        robot
            .chain()
            .forward_kinematics(&wrong_units, &mut pose)
            .is_err()
    );
}

#[test]
fn mixed_ik_and_servo_keep_linear_coordinates_and_velocity() {
    let robot = robot();
    let mut target = Transform3D::<f64>::identity();
    robot
        .chain()
        .forward_kinematics(&joints(0.4, 0.25), &mut target)
        .unwrap();
    let mut solution = joints(0.0, 0.1);
    let status = robot
        .chain()
        .inverse_kinematics_dls(
            &target,
            &joints(0.0, 0.1),
            &CuIkParams::default(),
            &mut solution,
        )
        .unwrap();
    assert!(matches!(status, CuIkStatus::Converged { .. }));
    assert!((solution[1].displacement().unwrap().get::<meter>() - 0.25).abs() < 1e-3);
    let mut velocity = [CuJointVelocity::default(); 2];
    robot
        .chain()
        .differential_ik(
            &joints(0.0, 0.2),
            &CuTwist6D::new(
                [0.0, 0.05, 0.0].map(Velocity::new::<meter_per_second>),
                [AngularVelocity::new::<radian_per_second>(0.0); 3],
                CuTwistFrame::Base,
            ),
            &CuDiffIkParams::default(),
            &mut velocity,
        )
        .unwrap();
    assert!(matches!(velocity[0], CuJointVelocity::Revolute(_)));
    assert!(matches!(velocity[1], CuJointVelocity::Prismatic(_)));
    assert!(velocity[1].si_value() > 0.0 && velocity[1].si_value() <= 0.2);
}

#[test]
fn geometry_retains_fixed_base_multiple_shapes_and_srdf_pairs() {
    let raw = CuRobotDescription::<2, 6>::from_urdf(URDF, urdf_import("base", "tool")).unwrap();
    let robot = robot();
    assert_eq!(
        robot.collision_links(),
        &["base", "pedestal", "arm", "arm", "slide", "tool"]
    );
    assert!(raw.collision_model().is_pair_enabled(2, 4));
    let skipping = CuRobotDescription::<2, 6>::from_urdf(
        URDF,
        urdf_import("base", "tool").with_collision(CuCollisionModelParams {
            skip_adjacent_links: true,
            ..collision()
        }),
    )
    .unwrap();
    assert!(!skipping.collision_model().is_pair_enabled(2, 4)); // adjacent frames 0 and 1
    assert!(!robot.collision_model().is_pair_enabled(2, 4));
    assert!(!robot.collision_model().is_pair_enabled(3, 4));
    assert!(!robot.collision_model().is_pair_enabled(4, 5)); // same rigid body
    let mut frames = [Transform3D::<f64>::identity(); 2];
    let mut pose = Transform3D::<f64>::identity();
    robot
        .chain()
        .forward_frames(&joints(0.0, 0.2), &mut frames, &mut pose)
        .unwrap();
    let mut shapes = [CuCollisionShape::default(); 6];
    robot
        .collision_model()
        .world_link_shapes(&frames, &mut shapes);
    if let CuCollisionShape::Sphere(s) = shapes[0] {
        assert_eq!(s.center[2].get::<meter>(), -1.0);
    } else {
        panic!("expected base sphere");
    }
    if let CuCollisionShape::Capsule(c) = shapes[2] {
        assert!((c.a[1].get::<meter>() - 0.15).abs() < 1e-12);
        assert!((c.b[1].get::<meter>() - 0.35).abs() < 1e-12);
    } else {
        panic!("expected cylinder enclosure");
    }
    let obstacles = [shapes[0]];
    let env = CuCollisionEnv::new(&obstacles);
    let mut report = CuCollisionReport::<8>::default();
    robot
        .collision_model()
        .check_configuration(robot.chain(), &joints(0.0, 0.2), &env, &mut report)
        .unwrap();
    assert!(!report.is_collision_free()); // base geometry participates in environment checks
}

#[test]
fn invalid_models_and_unsupported_features_fail_explicitly() {
    assert!(CuRobotDescription::<3, 6>::from_urdf(URDF, urdf_import("base", "tool")).is_err());
    assert!(CuRobotDescription::<2, 5>::from_urdf(URDF, urdf_import("base", "tool")).is_err());
    assert!(CuRobotDescription::<2, 6>::from_urdf(URDF, urdf_import("tool", "base")).is_err());
    for xml in [
        URDF.replace("radius=\"0.1\"", "radius=\"NaN\""),
        URDF.replace("xyz=\"0 0 2\"", "xyz=\"0 0 0\""),
        URDF.replace("type=\"prismatic\"", "type=\"floating\""),
        URDF.replace("<sphere radius=\"0.1\"/>", "<mesh filename=\"robot.stl\"/>"),
        URDF.replace("<axis xyz=\"1 0 0\"/>", "<mimic joint=\"swing\"/>"),
    ] {
        assert!(CuRobotDescription::<2, 6>::from_urdf(&xml, urdf_import("base", "tool")).is_err());
    }
    let mut robot = robot();
    for xml in [
        SRDF.replace("serial_demo", "other"),
        SRDF.replace("value=\"0.1\"", "value=\"8\""),
        SRDF.replace("<joint name=\"extension\" value=\"0.1\"/>", ""),
        SRDF.replace(
            "<chain base_link=\"base\" tip_link=\"tool\"/>",
            "<joint name=\"swing\"/>",
        ),
    ] {
        assert!(robot.apply_srdf(&xml, "manipulator").is_err());
    }
    let mut home = joints(0.0, 0.0);
    robot.write_named_state("home", &mut home).unwrap(); // failed updates are atomic
    assert_eq!(home[1].si_value(), 0.1);
}

#[test]
fn continuous_rotation_has_no_position_limit_and_empty_geometry_has_no_ghost_shape() {
    let urdf = r#"<robot name="continuous"><link name="base"/><link name="tip"/>
        <joint name="spin" type="continuous"><parent link="base"/><child link="tip"/>
        <limit velocity="2" effort="1"/></joint></robot>"#;
    let robot = CuRobotDescription::<1, 4>::from_urdf(urdf, urdf_import("base", "tip")).unwrap();
    let mut q = [Angle::new::<radian>(20.0).into()];
    assert!(robot.chain().is_within_limits(&q));
    robot.chain().clamp_to_limits(&mut q).unwrap();
    assert_eq!(q[0].angle().unwrap().get::<radian>(), 20.0);
    assert_eq!(robot.collision_model().shape_count(), 0);
    let mut report = CuCollisionReport::<0>::default();
    let env = CuCollisionEnv::EMPTY;
    robot
        .collision_model()
        .check_configuration(robot.chain(), &q, &env, &mut report)
        .unwrap();
    assert!(report.is_collision_free());
}

#[test]
fn folded_intermediate_rotations_preserve_the_full_transform() {
    for pitch in [
        0.7,
        core::f64::consts::FRAC_PI_2,
        -core::f64::consts::FRAC_PI_2,
    ] {
        let urdf = format!(
            r#"<robot name="folded">
        <link name="base"/><link name="a"/><link name="bracket"/><link name="b"/><link name="tip"/>
        <joint name="a" type="revolute"><parent link="base"/><child link="a"/>
          <origin xyz="0.1 0.2 0.3" rpy="0.2 0.3 0.4"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1"/></joint>
        <joint name="fixed" type="fixed"><parent link="a"/><child link="bracket"/>
          <origin xyz="0.2 -0.1 0.4" rpy="0.4 {pitch} -0.3"/></joint>
        <joint name="b" type="prismatic"><parent link="bracket"/><child link="b"/>
          <axis xyz="1 0 0"/><limit lower="0" upper="1" velocity="1"/></joint>
        <joint name="tip" type="fixed"><parent link="b"/><child link="tip"/>
          <origin xyz="0.1 0 0.2" rpy="-0.2 0.1 0.3"/></joint></robot>"#
        );
        let robot =
            CuRobotDescription::<2, 0>::from_urdf(&urdf, urdf_import("base", "tip")).unwrap();
        let transform = |xyz: [f64; 3], rpy: [f64; 3]| {
            Transform3D::<f64>::from_translation_euler_xyz(
                xyz.map(Length::new::<meter>),
                rpy.map(Angle::new::<radian>),
            )
        };
        let expected = transform([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
            .compose(transform([0.0; 3], [0.0, 0.0, 0.2]))
            .compose(transform([0.2, -0.1, 0.4], [0.4, pitch, -0.3]))
            .compose(transform([0.15, 0.0, 0.0], [0.0; 3]))
            .compose(transform([0.1, 0.0, 0.2], [-0.2, 0.1, 0.3]));
        let mut actual = Transform3D::<f64>::identity();
        robot
            .chain()
            .forward_kinematics(&joints(0.2, 0.15), &mut actual)
            .unwrap();
        for (a, b) in actual
            .to_matrix()
            .iter()
            .flatten()
            .zip(expected.to_matrix().iter().flatten())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

#[test]
fn topology_semantics_and_zero_capacity_reports_fail_safely() {
    let bad_cycle = URDF.replace(
        "<parent link=\"base\"/><child link=\"pedestal\"/>",
        "<parent link=\"tool\"/><child link=\"pedestal\"/>",
    );
    assert!(
        CuRobotDescription::<2, 6>::from_urdf(&bad_cycle, urdf_import("base", "tool")).is_err()
    );
    let duplicate = URDF.replace("</robot>", "<link name=\"arm\"/></robot>");
    assert!(
        CuRobotDescription::<2, 6>::from_urdf(&duplicate, urdf_import("base", "tool")).is_err()
    );
    let mut robot = robot();
    assert!(
        robot
            .apply_srdf(
                &SRDF.replace("link2=\"slide\"", "link2=\"typo\""),
                "manipulator"
            )
            .is_err()
    );
    let obstacles = [CuCollisionShape::Sphere(
        CuSphere::new(
            [0.0, 0.0, -1.0].map(Length::new::<meter>),
            Length::new::<meter>(0.1),
        )
        .unwrap(),
    )];
    let env = CuCollisionEnv::new(&obstacles);
    let mut report = CuCollisionReport::<0>::default();
    robot
        .collision_model()
        .check_configuration(robot.chain(), &joints(0.0, 0.2), &env, &mut report)
        .unwrap();
    assert!(!report.is_collision_free());
    assert_eq!(report.as_slice().len(), 0);
    assert!(report.total_contacts() > 0);
}

const INERTIAL_URDF: &str = r#"<?xml version="1.0"?>
<robot name="inertial_demo">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0.1"/>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <link name="l1">
    <inertial>
      <origin xyz="0.1 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>
  <link name="l1b">
    <inertial>
      <origin xyz="0 0.2 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>
  <link name="l2">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/><origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="10"/>
  </joint>
  <joint name="f1" type="fixed">
    <parent link="l1"/><child link="l1b"/><origin xyz="0.1 0 0"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="l1b"/><child link="l2"/><origin xyz="0 0.3 0"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.4" velocity="0.2" effort="10"/>
  </joint>
</robot>
"#;

#[test]
fn inertial_import_combines_fixed_links_into_moving_bodies() {
    let robot =
        CuRobotDescription::<2, 1>::from_urdf(INERTIAL_URDF, urdf_import("base", "l2")).unwrap();
    let links = robot.chain_inertia().links();
    let mass = |i: usize| links[i].mass.get::<kilogram>();
    let com = |i: usize| links[i].center_of_mass.map(|c| c.get::<meter>());
    let tensor = |i: usize| links[i].inertia.map(|v| v.get::<kilogram_square_meter>());
    // Body 0 folds l1 (2 kg at x = 0.1) and l1b (1 kg at (0.1, 0.2) in the
    // joint frame): mass-weighted centre plus parallel-axis-shifted tensors,
    // recombined about the shared centre of mass. The base link's 10 kg never
    // moves, so it contributes nothing.
    assert!((mass(0) - 3.0).abs() < 1e-12);
    let c0 = com(0);
    assert!((c0[0] - 0.1).abs() < 1e-9);
    assert!((c0[1] - 0.2 / 3.0).abs() < 1e-9);
    assert!(c0[2].abs() < 1e-12);
    let expected = [0.17 / 3.0, 0.045, 43.0 / 600.0, 0.0, 0.0, 0.0];
    for (k, want) in expected.iter().enumerate() {
        assert!(
            (tensor(0)[k] - want).abs() < 1e-9,
            "component {k}: got {}, want {want}",
            tensor(0)[k]
        );
    }
    // Body 1 is l2 alone, with an identity centre-of-mass origin.
    assert!((mass(1) - 0.5).abs() < 1e-12);
    assert!(com(1).iter().all(|v| v.abs() < 1e-12));
    for (k, got) in tensor(1).iter().enumerate() {
        let want = if k < 3 { 0.002 } else { 0.0 };
        assert!((got - want).abs() < 1e-12);
    }
    // A massless body stays valid: the fixture without `<inertial>` imports as zeros.
    let plain = CuRobotDescription::<2, 6>::from_urdf(URDF, urdf_import("base", "tool")).unwrap();
    for link in plain.chain_inertia().links() {
        assert_eq!(link.mass, Mass::new::<kilogram>(0.0));
        assert!(
            link.inertia
                .iter()
                .all(|v| *v == MomentOfInertia::new::<kilogram_square_meter>(0.0))
        );
    }
    // Negative mass is rejected rather than imported.
    let bad_mass = INERTIAL_URDF.replacen("<mass value=\"2.0\"/>", "<mass value=\"-2.0\"/>", 1);
    assert!(CuRobotDescription::<2, 1>::from_urdf(&bad_mass, urdf_import("base", "l2")).is_err());
}
