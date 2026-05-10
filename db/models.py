from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, JSON, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()


def new_id():
    return str(uuid.uuid4())


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=new_id)
    query = Column(Text, nullable=False)
    status = Column(String, default="pending")      # pending, running, completed, failed
    final_answer = Column(Text)
    provenance_map = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    events = relationship("ExecutionEvent", back_populates="job", cascade="all, delete")
    tool_calls = relationship("ToolCallLog", back_populates="job", cascade="all, delete")


class ExecutionEvent(Base):
    __tablename__ = "execution_events"

    id = Column(String, primary_key=True, default=new_id)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    sequence = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)     # routing, agent_start, agent_end, budget_violation
    input_hash = Column(String)
    output_hash = Column(String)
    input_data = Column(JSON)
    output_data = Column(JSON)
    latency_ms = Column(Float)
    token_count = Column(Integer)
    policy_violation = Column(Boolean, default=False)
    violation_detail = Column(Text)

    job = relationship("Job", back_populates="events")


class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"

    id = Column(String, primary_key=True, default=new_id)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    tool_name = Column(String, nullable=False)
    attempt = Column(Integer, default=1)            # 1, 2, or 3 (max 2 retries)
    input_data = Column(JSON)
    output_data = Column(JSON)
    latency_ms = Column(Float)
    accepted = Column(Boolean)
    rejection_reason = Column(Text)
    failure_mode = Column(String)                   # timeout, empty, malformed, none
    timestamp = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="tool_calls")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True, default=new_id)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    total_cases = Column(Integer)
    passed = Column(Integer)
    overall_score = Column(Float)
    category_scores = Column(JSON)
    dimension_scores = Column(JSON)

    results = relationship("EvalResult", back_populates="run", cascade="all, delete")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(String, primary_key=True, default=new_id)
    run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    job_id = Column(String, ForeignKey("jobs.id"))
    case_id = Column(String, nullable=False)
    category = Column(String, nullable=False)       # baseline, ambiguous, adversarial
    query = Column(Text, nullable=False)
    expected_answer = Column(Text)
    actual_answer = Column(Text)

    score_correctness = Column(Float)
    score_citation = Column(Float)
    score_contradiction = Column(Float)
    score_tool_efficiency = Column(Float)
    score_budget_compliance = Column(Float)
    score_critique_agreement = Column(Float)
    total_score = Column(Float)

    justification = Column(JSON)
    exact_prompts = Column(JSON)
    exact_tool_calls = Column(JSON)

    run = relationship("EvalRun", back_populates="results")


class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"

    id = Column(String, primary_key=True, default=new_id)
    eval_run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    original_prompt = Column(Text, nullable=False)
    proposed_prompt = Column(Text, nullable=False)
    diff = Column(Text, nullable=False)
    justification = Column(Text, nullable=False)
    status = Column(String, default="pending")      # pending, approved, rejected
    reviewed_at = Column(DateTime)
    proposed_at = Column(DateTime, default=datetime.utcnow)

    delta_results = relationship("RewriteDelta", back_populates="rewrite", cascade="all, delete")


class RewriteDelta(Base):
    __tablename__ = "rewrite_deltas"

    id = Column(String, primary_key=True, default=new_id)
    rewrite_id = Column(String, ForeignKey("prompt_rewrites.id"), nullable=False)
    case_id = Column(String, nullable=False)
    before_score = Column(Float)
    after_score = Column(Float)
    delta = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    rewrite = relationship("PromptRewrite", back_populates="delta_results")
