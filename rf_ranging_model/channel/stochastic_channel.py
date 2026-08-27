"""Compact Saleh-Valenzuela-style statistical indoor channel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import C0
from ..types import PropagationPath


@dataclass(frozen=True, slots=True)
class SalehValenzuelaConfig:
    cluster_arrival_rate_hz: float = 25.0e6
    ray_arrival_rate_hz: float = 180.0e6
    cluster_decay_s: float = 18.0e-9
    ray_decay_s: float = 7.0e-9
    cluster_count: int = 3
    rays_per_cluster: int = 5
    los_gain: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.cluster_arrival_rate_hz,
            self.ray_arrival_rate_hz,
            self.cluster_decay_s,
            self.ray_decay_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("arrival rates and decay constants must be positive")
        if self.cluster_count < 1 or self.rays_per_cluster < 1:
            raise ValueError("cluster_count and rays_per_cluster must be at least one")


def generate_saleh_valenzuela(
    distance_m: float,
    *,
    config: SalehValenzuelaConfig | None = None,
    seed: int | None = None,
    include_los: bool = True,
) -> list[PropagationPath]:
    """Generate reproducible clustered multipath after the geometric delay."""

    if distance_m <= 0.0:
        raise ValueError("distance_m must be greater than zero")
    cfg = config or SalehValenzuelaConfig()
    rng = np.random.default_rng(seed)
    geometric_delay = distance_m / C0
    paths: list[PropagationPath] = []
    if include_los:
        paths.append(PropagationPath(geometric_delay, complex(cfg.los_gain), "LOS"))

    cluster_delay = 0.0
    for cluster_index in range(cfg.cluster_count):
        if cluster_index > 0:
            cluster_delay += float(rng.exponential(1.0 / cfg.cluster_arrival_rate_hz))
        ray_delay = 0.0
        for ray_index in range(cfg.rays_per_cluster):
            if cluster_index == 0 and ray_index == 0 and include_los:
                continue
            if ray_index > 0:
                ray_delay += float(rng.exponential(1.0 / cfg.ray_arrival_rate_hz))
            excess_delay = cluster_delay + ray_delay
            envelope = np.exp(
                -0.5
                * (
                    cluster_delay / cfg.cluster_decay_s
                    + ray_delay / cfg.ray_decay_s
                )
            )
            magnitude = envelope * np.sqrt(rng.exponential())
            phase = rng.uniform(-np.pi, np.pi)
            paths.append(
                PropagationPath(
                    geometric_delay + excess_delay,
                    complex(magnitude * np.exp(1j * phase)),
                    f"cluster_{cluster_index}_ray_{ray_index}",
                )
            )

    paths.sort(key=lambda path: path.delay_s)
    return paths
