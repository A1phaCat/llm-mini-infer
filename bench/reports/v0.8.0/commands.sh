# Run with bash from the repository's root.
# The benchmark's commands, as run: bench/engines/drive.py started every server below, one at a time, on one GPU,
# from the GPU tests' image of each measured commit (llm-mini-infer in the image's venv; vLLM v0.33.0 and SGLang
# v0.5.23 in venvs of their own, pinned to their vendor images' freezes: freeze-diff.txt). See README.md.
set -eu
OUT=out  # the driver's output folder

# Hosts p1 and p2 ran this block at the first measured commit (README.md), whose bench/engines/arms.py left --config
# out of llm-mini-infer's command, so the driver added the image's configuration file to each of its arms, as the
# image's entrypoint passes it (the loop below). This tree's arms name --config themselves: run from here, OURS stays
# empty, and README.md's first deviation (llm-mini-infer's configuration file) does not arise.
OURS=()
# as run at the first measured commit:
# for arm in ours ours-nobi; do
#   OURS+=(--add-arg "$arm=--config" --add-arg "$arm=$LLM_MINI_INFER_CONFIG")
# done

# The GPU control at the start, middle and end of each host
python3 bench/engines/drive.py --out "$OUT" --family control --label start

# Gemma 4 31B FP8 with its assistant (k = 5); A1 on p1, the rest on p2
python3 bench/engines/drive.py --out "$OUT" --model gemma --family smoke "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model gemma --family A1 --rotate 0 "${OURS[@]}"
# the bound: this folder's bounds/bound-gemma-scene.json (2 x the median of the engines' A1 c=1 e2e p50, as e2e p99);
# a run that measures its own A1 writes its own bound by that rule (README.md, "Definitions")
python3 bench/engines/drive.py --out "$OUT" --model gemma --family A2 --rotate 1 --bound-file bench/reports/v0.8.0/bounds/bound-gemma-scene.json "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model gemma --family C --rotate 2 "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model gemma --family D --rotate 0 "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --family control --label middle
python3 bench/engines/drive.py --out "$OUT" --model gemma --family B "${OURS[@]}"

# Qwen3.8 27B FP8 with its MTP layer (k = 3); SGLang's arms did not start at 64 sequences (README.md)
python3 bench/engines/drive.py --out "$OUT" --model qwen --family smoke "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model qwen --family A1 --rotate 0 "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model qwen --family A2 --engines ours,vllm --rotate 1 --bound-file bench/reports/v0.8.0/bounds/bound-qwen-scene.json "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model qwen --family C --engines ours,vllm --rotate 2 "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model qwen --family D --engines ours,vllm "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --model qwen --family B "${OURS[@]}"
python3 bench/engines/drive.py --out "$OUT" --family control --label end

# Host p3 ran this block at the second measured commit (README.md), whose arms.py names --config itself, as this tree's
# does (the venv step copies SGLang's cuRAND headers into the CUDA toolkit's include folder)
python3 bench/engines/drive.py --out "$OUT" --family control --label start
python3 bench/engines/drive.py --out "$OUT" --model gemma --family smoke --arm ours --arm vllm-bi-nodg --arm sglang
python3 bench/engines/drive.py --out "$OUT" --model qwen --family smoke --engines sglang --arm sglang
python3 bench/engines/drive.py --out "$OUT" --model gemma --family C --arm vllm-bi-nodg
python3 bench/engines/drive.py --out "$OUT" --model gemma --family I
python3 bench/engines/drive.py --out "$OUT" --family control --label middle
python3 bench/engines/drive.py --out "$OUT" --model qwen --family A1 --engines sglang
python3 bench/engines/drive.py --out "$OUT" --model qwen --family C --engines sglang
python3 bench/engines/drive.py --out "$OUT" --model qwen --family D --engines sglang --skip-validity
python3 bench/engines/drive.py --out "$OUT" --model qwen --family A1 --engines ours,vllm --rotate 0
python3 bench/engines/drive.py --out "$OUT" --family control --label end

# Each family's client commands (bench/load/run.py, bench/structured/validity.py): python3 bench/engines/drive.py ... --dry-run

# The servers, one command per arm and commit, as the manifests record them (manifests/; the driver adds the environment)
# gemma ours, first commit (ok): added: --config $LLM_MINI_INFER_CONFIG; the image's configuration file (--config, as the image's entrypoint gives it), which arms.py leaves out
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --set model.path=RedHatAI/gemma-4-31B-it-FP8-block --set speculative.model=google/gemma-4-31B-it-assistant --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=5 --config $LLM_MINI_INFER_CONFIG
# gemma ours-nobi, first commit (ok): added: --config $LLM_MINI_INFER_CONFIG; the image's configuration file (--config, as the image's entrypoint gives it), which arms.py leaves out
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --set model.path=RedHatAI/gemma-4-31B-it-FP8-block --set speculative.model=google/gemma-4-31B-it-assistant --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=5 --set dev.runner.cuda.batch_invariant=false --config $LLM_MINI_INFER_CONFIG
# gemma sglang, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --attention-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-31B-it-assistant --speculative-eagle-topk 1 --speculative-num-steps=5 --speculative-num-draft-tokens 6 --mm-process-config '{"image":{"max_soft_tokens":280}}'
# gemma sglang-bi, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --attention-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-31B-it-assistant --speculative-eagle-topk 1 --speculative-num-steps=5 --speculative-num-draft-tokens 6 --mm-process-config '{"image":{"max_soft_tokens":280}}' --enable-deterministic-inference
# gemma vllm, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":5}'
# gemma vllm-bi, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_BATCH_INVARIANT VLLM_CACHE_ROOT
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":5}'
# gemma vllm-k1, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}'
# gemma vllm-k4, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
# qwen ours, first commit (ok): added: --config $LLM_MINI_INFER_CONFIG; the image's configuration file (--config, as the image's entrypoint gives it), which arms.py leaves out
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --set model.path=Qwen/Qwen3.8-27B-FP8 --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=3 --config $LLM_MINI_INFER_CONFIG
# qwen ours-nobi, first commit (ok): added: --config $LLM_MINI_INFER_CONFIG; the image's configuration file (--config, as the image's entrypoint gives it), which arms.py leaves out
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --set model.path=Qwen/Qwen3.8-27B-FP8 --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=3 --set dev.runner.cuda.batch_invariant=false --config $LLM_MINI_INFER_CONFIG
# qwen sglang, first commit (failed)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --trust-remote-code --attention-backend flashinfer --speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-num-steps=3 --speculative-num-draft-tokens 4 --max-mamba-cache-size 320 --enable-linear-replayssm-spec
# qwen sglang-bi, first commit (failed)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --trust-remote-code --attention-backend triton --speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-num-steps=3 --speculative-num-draft-tokens 4 --max-mamba-cache-size 320 --enable-linear-replayssm-spec --enable-deterministic-inference
# qwen vllm, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# qwen vllm-bi, first commit (unsupported)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_BATCH_INVARIANT VLLM_CACHE_ROOT
#   vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# qwen vllm-k1, first commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
# gemma ours, second commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --config $LLM_MINI_INFER_CONFIG --set model.path=RedHatAI/gemma-4-31B-it-FP8-block --set speculative.model=google/gemma-4-31B-it-assistant --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=5
# gemma ours-nobi, second commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --config $LLM_MINI_INFER_CONFIG --set model.path=RedHatAI/gemma-4-31B-it-FP8-block --set speculative.model=google/gemma-4-31B-it-assistant --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=5 --set dev.runner.cuda.batch_invariant=false
# gemma sglang, second commit (ok): the cuRAND headers of SGLang's venv (nvidia/cu13/include/curand*.h, the folder as found) copied into the CUDA toolkit's include folder: its speculative-sampling JIT kernel includes curand.h, which the GPU tests' image lacks (SGLang's own image is CUDA devel)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --attention-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-31B-it-assistant --speculative-eagle-topk 1 --speculative-num-steps=5 --speculative-num-draft-tokens 6 --mm-process-config '{"image":{"max_soft_tokens":280}}'
# gemma sglang-bi, second commit (ok): the cuRAND headers of SGLang's venv (nvidia/cu13/include/curand*.h, the folder as found) copied into the CUDA toolkit's include folder: its speculative-sampling JIT kernel includes curand.h, which the GPU tests' image lacks (SGLang's own image is CUDA devel)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --attention-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-31B-it-assistant --speculative-eagle-topk 1 --speculative-num-steps=5 --speculative-num-draft-tokens 6 --mm-process-config '{"image":{"max_soft_tokens":280}}' --enable-deterministic-inference
# gemma vllm, second commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":5}'
# gemma vllm-bi-nodg, second commit (ok): VLLM_USE_DEEP_GEMM=0: DeepGEMM off, so vLLM selects its next block-FP8 kernel (the arm with DeepGEMM, vLLM's default, runs beside it)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_BATCH_INVARIANT VLLM_CACHE_ROOT VLLM_USE_DEEP_GEMM
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":5}'
# qwen ours, second commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME CUTE_DSL_CACHE_DIR FLASHINFER_DISABLE_JIT FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH LLM_MINI_INFER_BUILD LLM_MINI_INFER_CONFIG LLM_MINI_INFER_KERNEL_PREBUILT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TOKENIZERS_PARALLELISM TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VIRTUAL_ENV
#   llm-mini-infer serve --config $LLM_MINI_INFER_CONFIG --set model.path=Qwen/Qwen3.8-27B-FP8 --set model.max_model_len=8184 --set scheduler.max_num_seqs=64 --set server.host=127.0.0.1 --set server.port=8000 --set speculative.num_speculative_tokens=3
# qwen sglang, second commit (ok): --max-running-requests 48 (SGLang's own value under speculative decoding), not 64: at 64 its fp32 mamba state (5 slots a request, ~49 GB) does not fit the ~41 GB left after the weights; a load at a higher concurrency queues in the server; the cuRAND headers of SGLang's venv (nvidia/cu13/include/curand*.h, the folder as found) copied into the CUDA toolkit's include folder: its speculative-sampling JIT kernel includes curand.h, which the GPU tests' image lacks (SGLang's own image is CUDA devel)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-running-requests 48 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --trust-remote-code --attention-backend flashinfer --speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-num-steps=3 --speculative-num-draft-tokens 4 --max-mamba-cache-size 240 --enable-linear-replayssm-spec
# qwen sglang-bi, second commit (ok): --max-running-requests 48 (SGLang's own value under speculative decoding), not 64: at 64 its fp32 mamba state (5 slots a request, ~49 GB) does not fit the ~41 GB left after the weights; a load at a higher concurrency queues in the server; the cuRAND headers of SGLang's venv (nvidia/cu13/include/curand*.h, the folder as found) copied into the CUDA toolkit's include folder: its speculative-sampling JIT kernel includes curand.h, which the GPU tests' image lacks (SGLang's own image is CUDA devel)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED SGLANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
#   sglang serve --model-path Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-running-requests 48 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --trust-remote-code --attention-backend triton --speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-num-steps=3 --speculative-num-draft-tokens 4 --max-mamba-cache-size 240 --enable-linear-replayssm-spec --enable-deterministic-inference
# qwen vllm, second commit (ok)
#   environment: CUDA_CACHE_PATH CUDA_HOME FLASHINFER_WORKSPACE_BASE HF_HOME HF_HUB_OFFLINE HOME LANG LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES PATH PYTHONUNBUFFERED TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
#   vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

# The vendor-image check (family Q, hosts q-vllm and q-sglang): each engine's own image (the manifests' `image`), run
# with the image's whole environment under the arm's variables and the image's own command; the clients from a folder
# holding this tree's files (DIR; the check ran the first measured commit's):
#   python3 bench/engines/drive.py --out "$OUT" --family control --label start
#   python3 bench/engines/drive.py --out "$OUT" --model gemma --family Q --engines vllm --engine-bin vllm=vllm --inherit-env --client-pythonpath DIR
#   python3 bench/engines/drive.py --out "$OUT" --model gemma --family Q --engines sglang --engine-bin sglang=sglang --inherit-env --client-pythonpath DIR
# gemma vllm, vendor image vllm/vllm-openai:v0.33.0@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3 (ok)
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-model-len 8184 --max-num-seqs 64 --gpu-memory-utilization 0.9 --enable-prefix-caching --limit-mm-per-prompt '{"image":16}' --seed 0 --enable-prompt-tokens-details --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}' --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":5}'
# gemma sglang, vendor image lmsysorg/sglang:v0.5.23-cu130-runtime@sha256:354700f4404ab0b8dcb493bc61d5d11e7c2a91f3a7babb4efe741cb75d4b51c0 (ok)
#   sglang serve --model-path RedHatAI/gemma-4-31B-it-FP8-block --host 127.0.0.1 --port 8000 --max-running-requests 64 --context-length 8184 --enable-metrics --enable-cache-report --random-seed 0 --grammar-backend llguidance --constrained-json-disable-any-whitespace --attention-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-31B-it-assistant --speculative-eagle-topk 1 --speculative-num-steps=5 --speculative-num-draft-tokens 6 --mm-process-config '{"image":{"max_soft_tokens":280}}'
