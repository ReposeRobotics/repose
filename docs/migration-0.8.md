# Migrating to Repose 0.8

Update all Repose crate dependencies together to `0.8.0` (or git tag `v0.8.0`).

`CuPlanStatus` adds `NoIkSolution`. Add this variant to exhaustive matches and
handle it as a final failure with no executable path. It means that none of the
bounded IK attempts reached the requested TCP pose; it does not prove the pose
is unreachable. Existing variants retain their encoding discriminants, and
joint request and path payload fields are unchanged.

Pose and Cartesian task setup is described in [Pose planning](pose-planning.md).
