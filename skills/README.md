# Repose skills for coding agents

These [Agent Skills](https://agentskills.io) teach a coding agent (Claude Code,
Codex, Cursor, Copilot, …) how to **use** Repose in your own project: which
crate and task to pick, the conventions every API shares, the RON config
shapes, and the mistakes that compile but misbehave.

They are for Repose users. The skills for working on Repose itself are in
[`.agents/skills/`](../.agents/skills/).

| Skill | Use it for |
|---|---|
| [`repose-getting-started`](repose-getting-started/SKILL.md) | Dependency setup, crate map, shared conventions; start here |
| [`repose-robot-config`](repose-robot-config/SKILL.md) | Preset, inline chain and collision model, URDF/SRDF import |
| [`repose-kinematics`](repose-kinematics/SKILL.md) | FK, Jacobians, DLS IK, differential IK, servo |
| [`repose-collision`](repose-collision/SKILL.md) | Collision models, planning scenes, attached objects, exclusions |
| [`repose-motion-planning`](repose-motion-planning/SKILL.md) | Planner contract, choosing a planner, pose and Cartesian planning |
| [`repose-trajectory-and-control`](repose-trajectory-and-control/SKILL.md) | Timing, blending, follower, joint controller, gripper |
| [`repose-copper-app`](repose-copper-app/SKILL.md) | RON graphs, bridges, MuJoCo simulation, sim-to-hardware |
| [`repose-policy-rl`](repose-policy-rl/SKILL.md) | ONNX RL and VLA policies |

## Install

Copy the skill directories into the skills folder your agent reads, in your
project or your home directory. For example, from your project root:

```sh
git clone --depth 1 https://github.com/ReposeRobotics/repose /tmp/repose
mkdir -p .claude/skills                 # Claude Code; other agents: .agents/skills
cp -r /tmp/repose/skills/repose-* .claude/skills/
```

Take the skills from the same tag as your Repose dependency, so they describe
the API you build against.

## Maintenance

The skills describe the public API, so a change to a public name, a config
key, a default, or a behavior they describe updates the affected skill in the
same PR. When a skill and the code disagree, the code wins.
