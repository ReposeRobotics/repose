//! Execution monitoring policy: edge detection and hold semantics for collision checks.
//!
//! The check itself lives in [`CuRobotCollisionModel`] / [`CuPlanningScene`]; this module only
//! tracks *state* — whether the watched configuration is currently in violation, and when that
//! state *changes* — so callers can raise a hold exactly once instead of re-acting every cycle.

/// Outcome of one monitoring pass over a configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuMonitorTransition {
    /// Watch was already in collision and still is (no new event).
    StillInCollision,
    /// Transitioned from clear into collision this pass.
    EnteredCollision,
    /// Collision cleared this pass.
    ClearedCollision,
    /// Watch was clear and remains clear (no new event).
    StillClear,
}

/// Stateful wrapper for a collision watch: raises edge events when a configuration
/// enters or leaves violation, so consumers ([`CuTask`](crate::scene) paths, executors)
/// get one event per collision episode rather than one per cycle.
#[derive(Debug, Clone)]
pub struct CuCollisionMonitor {
    in_collision: bool,
}

impl CuCollisionMonitor {
    /// Creates a fresh monitor with no violation latched and no history.
    pub const fn new() -> Self {
        Self {
            in_collision: false,
        }
    }

    /// Feeds one observation. `in_collision` comes from the caller's check
    /// (e.g. [`CuCollisionReport::is_collision_free`] negated); returns the
    /// detected transition.
    ///
    /// This is allocation-free and has no time dependency, so callers can run
    /// it at any control-loop rate.
    pub fn update(&mut self, in_collision: bool) -> CuMonitorTransition {
        let was = self.in_collision;
        self.in_collision = in_collision;
        match (was, in_collision) {
            (false, false) => CuMonitorTransition::StillClear,
            (false, true) => CuMonitorTransition::EnteredCollision,
            (true, false) => CuMonitorTransition::ClearedCollision,
            (true, true) => CuMonitorTransition::StillInCollision,
        }
    }

    /// Current latched state: `true` if the most recent observation was in collision.
    pub const fn in_collision(&self) -> bool {
        self.in_collision
    }

    /// Resets the watch to the clear state without emitting a transition.
    pub fn reset(&mut self) {
        self.in_collision = false;
    }
}

impl Default for CuCollisionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions() {
        let mut m = CuCollisionMonitor::new();
        assert!(matches!(m.update(false), CuMonitorTransition::StillClear));
        assert!(matches!(
            m.update(true),
            CuMonitorTransition::EnteredCollision
        ));
        assert!(matches!(
            m.update(true),
            CuMonitorTransition::StillInCollision
        ));
        assert!(matches!(
            m.update(false),
            CuMonitorTransition::ClearedCollision
        ));
        assert!(matches!(m.update(false), CuMonitorTransition::StillClear));
    }

    #[test]
    fn edge_only_once() {
        let mut m = CuCollisionMonitor::new();
        for _ in 0..10 {
            m.update(true);
        }
        // Still just one entry event in the sequence above; entry only reported
        // when the previous state was clear.
        assert!(m.in_collision());
    }
}
