"""Simple material presets for deterministic reflection studies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    relative_permittivity: float
    conductivity_s_per_m: float
    reflection_coefficient: complex


MATERIALS: dict[str, Material] = {
    "drywall": Material("drywall", 2.5, 0.02, -0.35 + 0.05j),
    "concrete": Material("concrete", 5.3, 0.12, -0.58 + 0.04j),
    "glass": Material("glass", 6.5, 0.01, -0.48 + 0.08j),
    "wood": Material("wood", 2.0, 0.01, -0.28 + 0.03j),
    "metal": Material("metal", 1.0, 1.0e7, -0.98 + 0.0j),
    "human": Material("human", 38.0, 1.5, -0.62 + 0.18j),
}


def get_material(name: str) -> Material:
    try:
        return MATERIALS[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(MATERIALS))
        raise KeyError(f"unknown material {name!r}; available: {available}") from exc
