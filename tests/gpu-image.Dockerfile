# The GPU tests' image (docs/development.md, "The GPU suite"): a published llm-mini-infer image, its layers unchanged,
# with what a cloud GPU host and the test suite need on top. Never a release: the image workflow's test-image job (a
# dispatch with test_image and publish on) publishes it as ghcr.io/<owner>/llm-mini-infer:test-gpu-<commit> only, and
# checks that the release's layers are its first ones. By hand, on an image already published, its commit's packages
# first (deploy/Dockerfile's test-debs stage, any registry):
#
#   docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target test-debs -t <debs> --push .
#   git archive --format=tar HEAD -- . ':(exclude)llm_mini_infer' | tar -x -C <dir>
#   docker buildx build --platform linux/amd64 -f tests/gpu-image.Dockerfile \
#       --build-arg TEST_BASE=ghcr.io/deftinfer/llm-mini-infer@sha256:<digest> --build-arg TEST_DEBS=<debs@digest> \
#       --build-arg UV_IMAGE=$(sed -n 's/^ARG UV_IMAGE=//p' deploy/Dockerfile) <dir>
#
# The context is the commit's tracked files (`git archive`), so nothing git ignores (results, models, private data)
# can enter it; llm_mini_infer/ is left out, and a link to the image's installed package takes its place, so the suite
# runs the release's code and the tests that read the sources read the release's. This file's own ignore file
# (gpu-image.Dockerfile.dockerignore) takes the place of the root's .dockerignore, which would keep everything but the
# package out, so the tree keeps it too. On top of the image: the packages deploy/Dockerfile's debs stage downloads on
# the release's own userland (an SSH server and rsync, and the build toolchain the suite's compiler tests need and
# serving does not, at the release build's versions), installed offline from TEST_DEBS's /debs, so this build reaches
# no apt mirror; and the dev group's packages from uv.lock (pytest, hypothesis, transformers, ...) beside the image's,
# none of which may change: the build fails if uv would remove or replace one.
ARG TEST_BASE
ARG TEST_DEBS
ARG UV_IMAGE

FROM ${UV_IMAGE} AS uv

FROM ${TEST_DEBS} AS debs

FROM ${TEST_BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
ARG DEBIAN_FRONTEND=noninteractive
# The packages, from the debs stage's /debs alone (the image keeps no apt lists: nothing else can be fetched), then
# the compiler's checks: CUDA 13.0's nvcc at the release's 13.0.88 (ptxas with it, cicc in libnvvm, and the runtime's,
# CCCL's and crt's headers FlashInfer's JIT includes), gcc and g++, and Python 3.12's headers (Triton's launcher). The
# release has no nvcc or ptxas of its own; ninja is the venv's (FlashInfer's dependency).
RUN --mount=from=debs,source=/debs,target=/debs \
    apt-get install -y --no-install-recommends /debs/*.deb \
 && dpkg-query -W openssh-server rsync >/dev/null \
 && /usr/local/cuda/bin/nvcc --version | grep -q 'V13.0.88' \
 && test -x /usr/local/cuda/nvvm/bin/cicc -a -x /usr/bin/gcc -a -x /usr/bin/g++ -a -x /opt/venv/bin/ninja \
 && test -f /usr/include/python3.12/Python.h
# The release turns FlashInfer's JIT off; the suite needs it, so an empty value (not "0") turns it on again
ENV FLASHINFER_DISABLE_JIT=
WORKDIR /opt/llm-mini-infer/test
COPY . .
# The image's own venv and interpreter (deploy/Dockerfile's deps stage), or uv would make a new venv; then neither a
# removal nor a replacement (uv's ` - ` and ` ~ ` lines) nor a new venv, and FlashInfer's cut still in place
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 uv sync --locked --inexact --no-install-project --only-group dev 2>&1 | tee /tmp/sync.log \
 && ! grep -E '^ [-~] |virtual environment' /tmp/sync.log \
 && test "$(ls /opt/venv/lib/python3.12/site-packages/flashinfer_jit_cache/providers/sm120f/jit_cache | wc -l)" = 3 \
 && rm /tmp/sync.log \
 && ln -s /opt/venv/lib/python3.12/site-packages/llm_mini_infer llm_mini_infer
