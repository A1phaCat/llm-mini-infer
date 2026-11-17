"""How deep MTP drafts at most (architecture.md, "Speculative decoding"): the verify window's limit, and each drafter's
in the CUDA runners. Light (no torch), so that the configuration can read them: `config.derive` refuses k above
`MAX_MTP_K` and takes the assistant's largest k from `MAX_GEMMA_MTP_K`. The runners import their own limits."""

from __future__ import annotations

MAX_MTP_K = 7  # any drafter: the verify window is 1 + k <= 8 tokens
MAX_GEMMA_MTP_K = MAX_MTP_K  # Gemma 4's assistant: attention takes rows of any length, so the window is its only limit
# Qwen's MTP layer, the CUDA runner on a GPU (its CPU references take more): the GDN replay kernel's window and fold fit
# 8 tokens of shared memory (kernels/csrc/gdn_mtp.cu)
MAX_CUDA_MTP_K = 3
