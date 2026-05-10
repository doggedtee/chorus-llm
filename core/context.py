from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class AgentID(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DECOMPOSITION = "decomposition"
    RETRIEVAL = "retrieval"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    COMPRESSION = "compression"
    META = "meta"


class SubTask(BaseModel):
    task_id: str
    description: str
    task_type: str                          # research, compute, lookup, summarize
    dependencies: List[str] = []            # list of task_ids that must complete first
    status: str = "pending"                 # pending, running, completed
    result: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    relevance_score: float
    used_for: Optional[str] = None          # which part of the answer this chunk supported


class Claim(BaseModel):
    text: str                               # the exact span of text
    confidence: float                       # 0.0 to 1.0
    flagged: bool = False
    flag_reason: Optional[str] = None
    source_agent: Optional[str] = None


class AgentOutput(BaseModel):
    agent_id: str
    output_text: str
    claims: List[Claim] = []
    chunks_used: List[str] = []             # chunk_ids referenced
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = 0


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: str
    source_chunk_id: Optional[str] = None


class SharedContext(BaseModel):
    """
    The single object passed between all agents.
    Agents read from this and write their outputs back into it.
    The orchestrator controls who writes when.
    """
    job_id: str
    original_query: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # decomposition agent fills this
    sub_tasks: List[SubTask] = []

    # retrieval agent fills this
    retrieved_chunks: List[RetrievedChunk] = []

    # each agent appends its output here
    agent_outputs: Dict[str, AgentOutput] = {}

    # critique agent fills this
    critiqued_claims: List[Claim] = []

    # synthesis agent fills this
    final_answer: Optional[str] = None
    provenance_map: List[ProvenanceEntry] = []

    # orchestrator fills this — routing log
    routing_log: List[Dict[str, Any]] = []

    # budget manager fills this
    token_usage: Dict[str, int] = {}        # {agent_id: tokens_used}
    budget_violations: List[str] = []       # agent_ids that violated budget

    # retrieval agent fills this — per sub-task Claude outputs
    task_outputs: Dict[str, str] = {}  # task_id → reasoning output

    # tool call results (latest per tool)
    tool_results: Dict[str, Any] = {}

    def add_routing_decision(self, from_agent: str, to_agent: str, reason: str):
        self.routing_log.append({
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def set_agent_output(self, agent_id: str, output: AgentOutput):
        self.agent_outputs[agent_id] = output

    def get_agent_output(self, agent_id: str) -> Optional[AgentOutput]:
        return self.agent_outputs.get(agent_id)

    def record_tokens(self, agent_id: str, count: int):
        self.token_usage[agent_id] = self.token_usage.get(agent_id, 0) + count

    def flag_budget_violation(self, agent_id: str):
        if agent_id not in self.budget_violations:
            self.budget_violations.append(agent_id)