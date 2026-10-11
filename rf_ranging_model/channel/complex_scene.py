"""Effective moving obstructions and two-bounce point scattering.

Lengths follow shared geometry; coefficients are not calibrated materials.
This model does not implement specular wall reflections or diffraction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..constants import C0
from ..types import PropagationPath
from .spatial import AttenuatingPanel, SpatialScene, vector3


@dataclass(frozen=True)
class MovingPanel:
    panel: AttenuatingPanel
    velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        vector3(self.velocity_m_s)

    def at(self, elapsed_s):
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        offset = vector3(self.velocity_m_s) * elapsed_s
        return replace(self.panel, lower_m=tuple(vector3(self.panel.lower_m) + offset),
                       upper_m=tuple(vector3(self.panel.upper_m) + offset))


@dataclass(frozen=True)
class ComplexScene:
    scene: SpatialScene
    moving_panels: tuple[MovingPanel, ...] = ()
    bounce_pairs: tuple[tuple[int, int], ...] = ()
    second_bounce_scale: float = 0.35

    def __post_init__(self):
        if not np.isfinite(self.second_bounce_scale) or not 0 <= self.second_bounce_scale <= 1:
            raise ValueError("second bounce scale must lie in [0, 1]")
        for first, second in self.bounce_pairs:
            if first == second or min(first, second) < 0 or max(first, second) >= len(self.scene.scatterers):
                raise ValueError("a second bounce requires two distinct existing scatterers")

    def paths(self, receiver_m, antenna, elapsed_s=0.0):
        current = replace(self.scene, panels=self.scene.panels + tuple(panel.at(elapsed_s) for panel in self.moving_panels))
        paths = current.paths(receiver_m, antenna)
        transmitter = vector3(current.transmitter_m)
        receiver = vector3(receiver_m) + vector3(antenna.offset_m)
        for first_index, second_index in self.bounce_pairs:
            first, second = current.scatterers[first_index], current.scatterers[second_index]
            vertices = (transmitter, vector3(first.position_m), vector3(second.position_m), receiver)
            lengths = [np.linalg.norm(end - start) for start, end in zip(vertices[:-1], vertices[1:])]
            if min(lengths) <= 1e-9:
                raise ValueError("successive scattering vertices must be distinct")
            transmission = np.prod([current.transmission(start, end) for start, end in zip(vertices[:-1], vertices[1:])])
            field = vector3(current.transmitter_polarization)
            field = field / np.linalg.norm(field)
            for start, end in zip(vertices[:-1], vertices[1:]):
                direction = (end - start) / np.linalg.norm(end - start)
                field = field - np.dot(field, direction) * direction
            receive_axis = vector3(antenna.polarization)
            coupling = np.dot(field, receive_axis / np.linalg.norm(receive_axis))
            length = float(sum(lengths))
            gain = transmission * coupling * first.coefficient * second.coefficient
            gain *= self.second_bounce_scale * current.amplitude_reference_m / length
            if abs(gain) > 1e-12:
                paths.append(PropagationPath(length / C0, complex(gain), f"double_{first_index}_{second_index}"))
        return paths


def bilateral_product(local, remote, local_mask, remote_mask):
    local, remote = np.asarray(local), np.asarray(remote)
    local_mask, remote_mask = np.asarray(local_mask), np.asarray(remote_mask)
    if local.shape != remote.shape or local.shape != local_mask.shape or local.shape != remote_mask.shape:
        raise ValueError("bilateral samples and masks must have matching shapes")
    if local_mask.dtype != bool or remote_mask.dtype != bool:
        raise ValueError("bilateral quality masks must be boolean")
    mask = local_mask & remote_mask & np.isfinite(local) & np.isfinite(remote)
    return np.where(mask, local, 0) * np.where(mask, remote, 0), mask
