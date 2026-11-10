-- 0002: index for the decision log's update path.
-- DecisionLog.update() runs UPDATE plan_decisions ... WHERE plan_id = ? AND action = ?
-- after every approve / reject; without an index that is a scan of the whole log.

CREATE INDEX IF NOT EXISTS plan_decisions_plan_action_idx ON plan_decisions (plan_id, action);
