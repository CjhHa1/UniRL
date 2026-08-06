"""Rollout engines over the canonical ``Sample`` request type.

``synchronous.py`` records the worker-side `Sample -> Sample` contracts.
``asynchronous.py`` contains the batch-granular driver engine; agentic scheduling
lives in ``unirl.rollout.manager``.

This module remains empty so consumers import the required layer directly.
"""
