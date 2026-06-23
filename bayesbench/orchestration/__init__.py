"""orchestration — shared, backend-agnostic infrastructure for multi-turn belief-updating environments.

Pure Python (no torch / vLLM imports) so it is importable anywhere. Generation
backends are injected as callables by each task. See README for how to add an
environment.
"""
