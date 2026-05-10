from typing import Dict
from core.context import SharedContext

# max tokens each agent is allowed to use per job
AGENT_BUDGETS: Dict[str, int] = {
    "orchestrator":  2000,
    "decomposition": 3000,
    "retrieval":     6000,
    "critique":      4000,
    "synthesis":     5000,
    "compression":   2000,
    "meta":          4000,
}


class BudgetManager:
    def __init__(self, context: SharedContext):
        self.context = context

    def get_budget(self, agent_id: str) -> int:
        """Return the max token budget for an agent."""
        return AGENT_BUDGETS.get(agent_id, 2000)

    def get_used(self, agent_id: str) -> int:
        """Return how many tokens this agent has used so far."""
        return self.context.token_usage.get(agent_id, 0)

    def get_remaining(self, agent_id: str) -> int:
        """Return how many tokens this agent still has left."""
        return self.get_budget(agent_id) - self.get_used(agent_id)

    def can_proceed(self, agent_id: str, estimated_tokens: int) -> bool:
        """Check if agent has enough budget before adding more context."""
        return self.get_remaining(agent_id) >= estimated_tokens

    def consume(self, agent_id: str, tokens_used: int):
        """
        Record token usage for an agent.
        If it exceeds budget, flag it as a policy violation.
        """
        self.context.record_tokens(agent_id, tokens_used)

        if self.get_used(agent_id) > self.get_budget(agent_id):
            self.context.flag_budget_violation(agent_id)
            print(
                f"[POLICY VIOLATION] {agent_id} exceeded budget: "
                f"{self.get_used(agent_id)} / {self.get_budget(agent_id)} tokens"
            )

    def needs_compression(self, agent_id: str, threshold: float = 0.85) -> bool:
        """Return True if agent has used more than 85% of its budget."""
        used = self.get_used(agent_id)
        budget = self.get_budget(agent_id)
        return (used / budget) >= threshold if budget > 0 else False

    def summary(self) -> Dict[str, dict]:
        """Return a summary of budget usage for all agents."""
        result = {}
        for agent_id, budget in AGENT_BUDGETS.items():
            used = self.get_used(agent_id)
            result[agent_id] = {
                "budget": budget,
                "used": used,
                "remaining": budget - used,
                "violation": agent_id in self.context.budget_violations,
            }
        return result