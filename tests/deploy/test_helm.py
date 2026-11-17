"""llm-mini-infer's Helm chart (deploy/helm/llm-mini-infer), rendered with its own values and with an example
values file (tests/data/chart-values.yaml): the configuration file it renders, the probes and the drain, secrets by
name, the scheduling, the image, and the label guard: the chart refuses pod labels that carry every label of a selector
named in `notSelectedBy`, such as one another workload uses to find pods and send them traffic. helm is the
repository's pinned copy (`uv run tools/fetch_tools.py helm`, into tools/bin/); skipped without it.
"""

from __future__ import annotations

import pytest
import yaml

from ..support.helm import HELM_BIN, VALUES, kind, render
from ..support.slim_image import IMAGE_REPOSITORY, version

pytestmark = pytest.mark.skipif(HELM_BIN is None, reason="no helm in tools/bin: uv run tools/fetch_tools.py helm")

MODEL_SET = "model.id=Qwen/Qwen3.8-27B-FP8"  # the chart renders nothing without a model

# The selectors of tests/data/chart-values.yaml's notSelectedBy
ENGINE_SELECTOR = {"app": "llm", "component": "engine"}
GATEWAY_SERVICE = {"app": "llm", "component": "gateway"}


def matches(selector: dict, labels: dict) -> bool:
    return all(labels.get(k) == v for k, v in selector.items())


def test_the_pods_carry_no_selector_they_must_not_match() -> None:
    docs = render("llm-mini-infer", VALUES)
    pod = kind(docs, "Deployment")[0]["spec"]["template"]["metadata"]["labels"]
    assert pod["app"] == "llm"  # the values' podLabels, one of each selector's two labels
    assert not matches(ENGINE_SELECTOR, pod) and not matches(GATEWAY_SERVICE, pod)


@pytest.mark.parametrize("labels", [["podLabels.component=engine"], ["podLabels.component=gateway"]])
def test_the_chart_refuses_labels_a_gateway_would_adopt(labels: list[str]) -> None:
    with pytest.raises(RuntimeError, match="would adopt the pods"):
        render("llm-mini-infer", VALUES, *labels)


def test_the_deployment_renders_the_values(tmp_path) -> None:
    from llm_mini_infer import config

    values = yaml.safe_load(VALUES.read_text())
    want = values["config"]
    docs = render("llm-mini-infer", VALUES)
    spec = kind(docs, "Deployment")[0]["spec"]["template"]["spec"]
    c = spec["containers"][0]
    (tmp_path / "config.yaml").write_text(kind(docs, "ConfigMap")[0]["data"]["config.yaml"])
    cfg = config.load(str(tmp_path / "config.yaml"), {})
    # the configuration file: model.path from model.id (the served name is the repo id), the rest as given
    assert cfg.model.path == values["model"]["id"] and cfg.model.served_model_name is None
    assert cfg.speculative.model == want["speculative"]["model"]
    assert cfg.speculative.num_speculative_tokens is None  # llm-mini-infer's own k
    assert (cfg.images.max_soft_tokens, cfg.images.max_per_request) == (
        want["images"]["max_soft_tokens"], want["images"]["max_per_request"])
    assert cfg.images.min_pixels is None and cfg.images.max_pixels is None  # Gemma 4 refuses pixel bounds
    assert cfg.structured_output.backend == want["structured_output"]["backend"]
    assert cfg.structured_output.disable_any_whitespace is want["structured_output"]["disable_any_whitespace"]
    assert cfg.model.max_model_len == want["model"]["max_model_len"] and cfg.runner.kind == want["runner"]["kind"]
    assert cfg.scheduler.seed == want["scheduler"]["seed"]
    assert cfg.server.keep_alive_s == want["server"]["keep_alive_s"]
    assert cfg.runner.cuda.memory_utilization == want["runner"]["cuda"]["memory_utilization"]
    assert cfg.scheduler.max_num_seqs == want["scheduler"]["max_num_seqs"]
    # probes: readiness /health, liveness /live (never /health: 503 while draining), startup /health
    assert c["readinessProbe"]["httpGet"]["path"] == "/health"
    assert c["livenessProbe"]["httpGet"]["path"] == "/live"
    assert c["startupProbe"]["httpGet"]["path"] == "/health"
    assert spec["terminationGracePeriodSeconds"] > cfg.server.drain_grace_s + cfg.server.drain_timeout_s
    # secrets by name; the chart creates none
    env = {e["name"]: e for e in c["env"]}
    assert env["VLLM_API_KEY"]["valueFrom"]["secretKeyRef"] == {"name": "llm-secrets", "key": "VLLM_API_KEY"}
    assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {"name": "llm-secrets", "key": "HF_TOKEN"}
    assert not kind(docs, "Secret") and not kind(docs, "SealedSecret")
    # one GPU, CPU limits for the derived defaults, the shm arena, the cache volume
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1 and c["resources"]["limits"]["cpu"] == "16"
    vols = {v["name"]: v for v in spec["volumes"]}
    assert vols["shm"]["emptyDir"]["medium"] == "Memory" and "emptyDir" in vols["cache"]
    assert spec["runtimeClassName"] == "nvidia"
    # never beside the pods named in avoidPodsWithLabels, nor beside another llm-mini-infer
    terms = spec["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert {"component": "other-engine"} in [t["labelSelector"]["matchLabels"] for t in terms]
    gpu = spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert gpu[0]["matchExpressions"][0]["values"] == ["NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"]
    headless = [s for s in kind(docs, "Service") if s["spec"].get("clusterIP") == "None"]
    assert headless and headless[0]["metadata"]["name"] == "llm-mini-infer-headless"
    assert kind(docs, "PodDisruptionBudget") and kind(docs, "ServiceMonitor")
    # the image at the chart's release (its appVersion, pyproject's version), through a pull secret;
    # as the image's non-root user, with a mounted claim's group
    assert c["image"] == f"ghcr.io/example/llm-mini-infer:{version()}"
    assert spec["imagePullSecrets"] == [{"name": "registry-credentials"}]
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["runAsUser"] == spec["securityContext"]["fsGroup"] == 10001


def test_the_default_image_is_the_release_and_a_tag_overrides_it() -> None:
    """The chart's default image tag is its appVersion, pyproject's version: the image a release (git tag v<version>)
    publishes; a commit's tag overrides it."""
    image = lambda docs: kind(docs, "Deployment")[0]["spec"]["template"]["spec"]["containers"][0]["image"]  # noqa: E731
    default = image(render("llm-mini-infer", None, MODEL_SET))
    assert default == f"{IMAGE_REPOSITORY}:{version()}"
    tagged = image(render("llm-mini-infer", None, MODEL_SET, "image.tag=1a2b3c4"))
    assert tagged == f"{IMAGE_REPOSITORY}:1a2b3c4"


def test_the_chart_requires_a_model() -> None:
    with pytest.raises(RuntimeError, match=r"model\.id is required"):
        render("llm-mini-infer")


def test_avoided_pods_are_avoided_in_every_namespace() -> None:
    """Another engine claims its node's GPU whichever namespace it runs in: each avoid term carries an empty
    namespaceSelector, which matches every namespace (a term without one sees only llm-mini-infer's namespace).
    llm-mini-infer's own replicas are kept apart within the release."""
    docs = render("llm-mini-infer", VALUES)
    spec = kind(docs, "Deployment")[0]["spec"]["template"]["spec"]
    terms = spec["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    avoid = [t for t in terms if t["labelSelector"]["matchLabels"] == {"component": "other-engine"}]
    own = [t for t in terms if t not in avoid]
    assert len(avoid) == 1 and avoid[0]["namespaceSelector"] == {} and "namespaces" not in avoid[0]
    assert avoid[0]["topologyKey"] == "kubernetes.io/hostname"
    assert len(own) == 1 and own[0]["labelSelector"]["matchLabels"]["app.kubernetes.io/name"] == "llm-mini-infer"
    assert "namespaceSelector" not in own[0]
    # with no terms configured, only llm-mini-infer's own
    spec = kind(render("llm-mini-infer", None, MODEL_SET), "Deployment")[0]["spec"]["template"]["spec"]
    terms = spec["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert len(terms) == 1 and "namespaceSelector" not in terms[0]


def test_a_grace_period_shorter_than_the_drain_is_refused() -> None:
    with pytest.raises(RuntimeError, match="terminationGracePeriodSeconds"):
        render("llm-mini-infer", VALUES, "terminationGracePeriodSeconds=60")


def test_the_grace_period_needs_20_s_beyond_the_drain() -> None:
    """The processes' stop after the drain: the frontends' 5 s, the engine's 10 s and 5 s more after a kill
    (`launcher._shutdown`). The chart's defaults drain for 20 + 30 s."""
    with pytest.raises(RuntimeError, match="by more than 20 s"):
        render("llm-mini-infer", None, MODEL_SET, "terminationGracePeriodSeconds=70")
    assert kind(render("llm-mini-infer", None, MODEL_SET, "terminationGracePeriodSeconds=71"), "Deployment")
