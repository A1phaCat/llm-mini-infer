"""A recorded /metrics page, the other engine's (tests/api/test_hostcheck.py, tests/repo/test_bench_kit.py)."""

from __future__ import annotations

# vLLM's page: its spec-decode counters (prometheus_client adds _total, and _created samples), no
# llm_mini_infer:
VLLM_METRICS = """# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} %(drafts)s
vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 1.7e+09
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} %(p0)s
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} %(p1)s
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} %(p2)s
vllm:generation_tokens_total{engine="0",model_name="m"} %(gen)s
"""
