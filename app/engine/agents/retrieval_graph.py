"""LangGraph-shaped retrieval workflow on the in-memory / hybrid stack.

nodes: route -> plan -> search -> grade_relevance -> (rewrite | answer) -> verify -> respond
Hard iteration cap. Destructive moves stay on the MCP tools with HITL.
"""

from app.agent.retrieval_loop import RetrievalAgent

__all__ = ["RetrievalAgent"]
