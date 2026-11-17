"""The host check's synthetic workload (`llm_mini_infer.hostcheck`), and the image tests' test-pattern images: one
image, a long system prompt and a strict JSON schema per request, with every part generated here.

The task is a made-up scene description: count the objects in a photo, describe up to six of them and the setting.
Its parts:
- **Images** (`pattern`, `image_jpeg`): a flat colour and 12 rectangles drawn from a seed, at two sizes: 1280 × 720
  (landscape) and 720 × 1280 (portrait), a share `tall_share` of them portrait (default `TALL_SHARE`), spread out. Both
  resize to the same number of patches and image tokens at each of Gemma 4's budgets, the portrait one's grid
  transposed: 264 tokens at its default 280 (36 × 66 against 66 × 36). Every index gives a different image, so no
  request hits an image cache.
- **The schema** (`scene_schema`): an integer `count`, a list `objects` of at most `MAX_OBJECTS` objects and one
  `setting`. An object has an `id`, the nullable list attributes `OBJECT` (values from fixed sets), a `size` from
  `SIZE` and a boolean `occluded`; the setting has the list attributes `SETTING`. Every object refuses other keys and
  requires all of its own, in order, as pydantic writes them, with `$defs` for the object and the setting.
- **The prompt** (`system_prompt`): an instruction for that schema, padded with numbered rules up to `target` of the
  model's tokens (default `PROMPT_TOKENS`).
- **The user turn** (`USER_TEXT`): one short sentence beside the image.

A test pattern holds few objects, so the answers are short: the workload measures the host and the server, not a
model's answers to real photos. `WORKLOAD_VERSION` names this definition in a reference's labels, so a reference
measured on another one never matches.

`write_inputs` writes the workload as the host check's mounted inputs folder, an example of that folder's layout.
"""

from __future__ import annotations

import io
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

WORKLOAD_VERSION = 2  # the definition below: sizes, schema, prompt and user text
# (name, values, the list's maximum length, at least one value when not null)
OBJECT = (
    ("kind", ("vehicle", "furniture", "plant", "tool", "container", "sign", "appliance", "building"), 2, True),
    ("colour", ("red", "orange", "yellow", "green", "blue", "purple", "black", "white", "grey", "brown"), 3, True),
    ("shape", ("round", "square", "rectangular", "triangular", "irregular"), 2, True),
    ("position", ("left", "centre", "right", "top", "bottom"), 2, True),
    ("material", ("metal", "wood", "plastic", "glass", "fabric", "stone", "paper"), 3, False),
    ("condition", ("new", "worn", "damaged"), 1, False),
    ("texture", ("smooth", "rough", "shiny", "matte", "patterned"), 2, False),
    ("parts", ("handle", "wheel", "door", "lid", "screen", "label", "cable"), 4, False),
)
SIZE = ("tiny", "small", "medium", "large")
SETTING = (
    ("place", ("indoors", "street", "park", "shop", "kitchen", "office", "workshop"), 2, True),
    ("light", ("daylight", "dusk", "night", "artificial"), 2, False),
    ("walls", ("painted", "brick", "tiled", "bare"), 2, False),
)
MAX_OBJECTS = 6
SIZES = ((1280, 720), (720, 1280))  # landscape, portrait
TALL_SHARE = 0.25  # the share of portrait images
PROMPT_TOKENS = 2048  # the system prompt's length in the model's tokens
USER_TEXT = "Describe this photo."
SCHEMA_NAME = "scene"


def pattern(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), tuple(rng.randrange(256) for _ in range(3)))
    for _ in range(12):
        x, y = rng.randrange(w), rng.randrange(h)
        img.paste(tuple(rng.randrange(256) for _ in range(3)), (x, y, min(w, x + rng.randrange(1, w)),
                                                                min(h, y + rng.randrange(1, h))))
    return img


def image_size(i: int, tall_share: float = TALL_SHARE) -> tuple[int, int]:
    """Image `i`'s (width, height): with `tall_share` p/q in lowest terms, portrait for p indices in every q, spread
    out."""
    if not 0 <= tall_share <= 1:
        raise ValueError(f"the share of tall images is between 0 and 1, got {tall_share}")
    f = Fraction(tall_share).limit_denominator(1000)
    return SIZES[1] if (i * f.numerator) % f.denominator < f.numerator else SIZES[0]


def image_jpeg(i: int, tall_share: float = TALL_SHARE) -> bytes:
    buf = io.BytesIO()
    pattern(*image_size(i, tall_share), seed=i).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _attribute(name: str, values: tuple[str, ...], max_items: int, nonempty: bool) -> dict[str, Any]:
    arr: dict[str, Any] = {"items": {"enum": list(values), "type": "string"}, "maxItems": max_items}
    if nonempty:
        arr["minItems"] = 1
    arr["type"] = "array"
    return {"anyOf": [arr, {"type": "null"}], "title": name.capitalize()}


def _object(title: str, props: dict[str, Any]) -> dict[str, Any]:
    return {"additionalProperties": False, "properties": props, "required": list(props), "title": title,
            "type": "object"}


def scene_schema() -> dict[str, Any]:
    """The workload's strict JSON schema (module docstring), as a pydantic model with `$defs` writes it."""
    obj = _object("Object", {"id": {"title": "Id", "type": "integer"},
                             **{n: _attribute(n, v, m, ne) for n, v, m, ne in OBJECT},
                             "size": {"enum": list(SIZE), "title": "Size", "type": "string"},
                             "occluded": {"title": "Occluded", "type": "boolean"}})
    setting = _object("Setting", {n: _attribute(n, v, m, ne) for n, v, m, ne in SETTING})
    top = _object("Scene", {"count": {"title": "Count", "type": "integer"},
                            "objects": {"items": {"$ref": "#/$defs/Object"}, "maxItems": MAX_OBJECTS,
                                        "title": "Objects", "type": "array"},
                            "setting": {"$ref": "#/$defs/Setting"}})
    return {"$defs": {"Object": obj, "Setting": setting},
            "description": "A synthetic description of one photo: the objects in it and its setting.", **top}


def system_prompt(count_tokens: Any, target: int = PROMPT_TOKENS) -> str:
    """An instruction for `scene_schema()`, padded with numbered rules until `count_tokens(text)` (the model's
    tokenizer) reaches `target`."""
    lines = ["You describe the objects in one photo. Answer with one JSON object and nothing else: count, the number "
             f"of objects you can see; objects, up to {MAX_OBJECTS} of them, the most visible first; setting, where "
             "the photo was taken. Give each object an id from 0, the attributes below, its size and whether it is "
             "occluded. Each attribute holds a list of values from its set, [] when the object visibly lacks it "
             "(where the attribute allows that), or null when the photo does not show it well enough to judge.", "",
             "Object attributes:"]
    for n, v, m, ne in OBJECT:
        lines.append(f"- {n}: {int(ne)} to {m} of {', '.join(v)}.")
    lines.append(f"- size: one of {', '.join(SIZE)}.")
    lines.append("- occluded: true when another object hides part of it.")
    lines += ["", "Setting attributes:"]
    for n, v, m, ne in SETTING:
        lines.append(f"- {n}: {int(ne)} to {m} of {', '.join(v)}.")
    lines += ["", "Rules:"]
    names = [n for n, *_ in OBJECT + SETTING]
    k = 0
    while count_tokens("\n".join(lines)) < target:
        a, b = names[k % len(names)], names[(k * 7 + 3) % len(names)]
        lines.append(f"{k + 1}. When the photo is dark, blurred or cropped, judge {a} only from what is clearly "
                     f"visible, and prefer null to a guess; never infer {a} from {b} or from the objects around it.")
        k += 1
    return "\n".join(lines)


def write_inputs(folder: Path, count: int, count_tokens: Any, prompt_tokens: int = PROMPT_TOKENS,
                 tall_share: float = TALL_SHARE) -> None:
    """The workload as the host check's mounted inputs folder: `frames/` (`count` images), `prompt.md`, `schema.json`
    and `user.txt`."""
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / "frames" / f"{i:06d}.jpg").write_bytes(image_jpeg(i, tall_share))
    (folder / "prompt.md").write_text(system_prompt(count_tokens, prompt_tokens))
    (folder / "schema.json").write_text(json.dumps(scene_schema(), indent=1))
    (folder / "user.txt").write_text(USER_TEXT + "\n")
