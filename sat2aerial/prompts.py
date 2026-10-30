from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSpec:
    name: str
    camera: str
    off_nadir_deg: int


VIEWS: dict[str, ViewSpec] = {
    "nadir": ViewSpec(
        name="nadir",
        camera="high-resolution near-nadir aerial photograph taken by a drone with the camera pointing straight down",
        off_nadir_deg=0,
    ),
    "oblique15": ViewSpec(
        name="oblique15",
        camera="high-resolution aerial photograph taken by a drone looking down at roughly 15 degrees off nadir",
        off_nadir_deg=15,
    ),
    "oblique30": ViewSpec(
        name="oblique30",
        camera="high-resolution oblique aerial photograph taken by a drone looking down at roughly 30 degrees off nadir",
        off_nadir_deg=30,
    ),
    "oblique45": ViewSpec(
        name="oblique45",
        camera="high-resolution oblique aerial photograph taken by a drone looking down at roughly 45 degrees off nadir",
        off_nadir_deg=45,
    ),
}


def build_prompt(
    view: str = "nadir",
    style_refs: int = 0,
    altitude_m: int = 120,
    season: str = "summer",
    lighting: str = "clear midday sunlight",
    weather: str = "clear",
    extra: str = "",
) -> str:
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}; choose from {sorted(VIEWS)}")
    spec = VIEWS[view]
    geometry = (
        "Preserve the exact geometry of <ref_image_0>: keep every road, building footprint, "
        "park, water body and field in the same position and orientation. Do not add, remove, "
        "move or resize structures. The result must depict the same real place."
    )
    style = ""
    if style_refs > 0:
        refs = ", ".join(f"<ref_image_{i + 1}>" for i in range(style_refs))
        style = (
            f" Match the camera characteristics, colour grading, texture, sharpness, noise level "
            f"and lighting of {refs}, which are real drone photographs. The output should look like "
            f"it came from the same aircraft and sensor as {refs}."
        )
    prompt = (
        f"A photorealistic {spec.camera}, altitude about {altitude_m} metres, "
        f"{lighting}, {weather}, season: {season}. {geometry}{style} "
        "Render only the appearance of an aerial photograph, not a map, diagram or illustration. "
        "No text, labels, borders, watermarks or UI overlays."
    )
    if extra:
        prompt = f"{prompt} {extra.strip()}"
    return prompt


VARIANT_VIEWS: dict[str, str] = {
    "nadir": "nadir",
    "oblique15": "oblique15",
    "oblique30": "oblique30",
    "oblique45": "oblique45",
}


def parse_variants(raw: str) -> list[tuple[str, str, int]]:
    variants: list[tuple[str, str, int]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        base = token
        style_refs = 0
        if token.endswith("_style"):
            base = token[: -len("_style")]
            style_refs = 1
        if "+style" in token:
            base, count = token.split("+style", 1)
            style_refs = int(count) if count else 1
        if base not in VARIANT_VIEWS:
            raise ValueError(f"Unknown variant {token!r}")
        variants.append((token, VARIANT_VIEWS[base], style_refs))
    if not variants:
        raise ValueError("No variants selected")
    return variants
