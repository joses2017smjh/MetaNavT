"""Retrieval primitives used by both the production HybridRetriever and the bench harness."""

from app.retrieval.fuse import reciprocal_rank_fusion
from app.retrieval.router import QueryRouter, RouteType

__all__ = ["reciprocal_rank_fusion", "QueryRouter", "RouteType"]
