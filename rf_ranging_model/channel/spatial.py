"""Shared 3-D point-scatterer geometry and finite attenuating panels.

This is a single-bounce effective scattering model, not a material-calibrated
ray tracer. All antenna paths are derived from one scene and actual positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import C0
from ..types import PropagationPath


def vector3(value):
    array = np.asarray(value, float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("expected a finite three-dimensional vector")
    return array


def unit_vector(value):
    array = vector3(value)
    length = float(np.linalg.norm(array))
    if length <= 1e-12:
        raise ValueError("a direction must have nonzero length")
    return array / length


@dataclass(frozen=True)
class SpatialAntenna:
    offset_m: tuple[float, float, float]
    polarization: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        vector3(self.offset_m)
        unit_vector(self.polarization)


@dataclass(frozen=True)
class PointScatterer:
    position_m: tuple[float, float, float]
    coefficient: complex
    polarization_rotation_rad: float = 0.0

    def __post_init__(self):
        vector3(self.position_m)
        if not np.isfinite(self.coefficient) or not np.isfinite(self.polarization_rotation_rad):
            raise ValueError("scattering parameters must be finite")


@dataclass(frozen=True)
class AttenuatingPanel:
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]
    attenuation_db: float = 0.0
    opaque: bool = False

    def __post_init__(self):
        if np.any(vector3(self.upper_m) <= vector3(self.lower_m)):
            raise ValueError("panel bounds must have positive extent")
        if not np.isfinite(self.attenuation_db) or self.attenuation_db < 0:
            raise ValueError("attenuation must be nonnegative and finite")

    def intersects(self, start_m, end_m):
        start, end = vector3(start_m), vector3(end_m)
        direction = end - start
        near, far = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < self.lower_m[axis] or start[axis] > self.upper_m[axis]:
                    return False
            else:
                crossings = sorted(((self.lower_m[axis] - start[axis]) / direction[axis],
                                    (self.upper_m[axis] - start[axis]) / direction[axis]))
                near, far = max(near, crossings[0]), min(far, crossings[1])
                if near > far:
                    return False
        return far > 1e-10 and near < 1 - 1e-10


@dataclass(frozen=True)
class SpatialScene:
    transmitter_m: tuple[float, float, float]
    scatterers: tuple[PointScatterer, ...] = ()
    panels: tuple[AttenuatingPanel, ...] = ()
    transmitter_polarization: tuple[float, float, float] = (0.0, 0.0, 1.0)
    amplitude_reference_m: float = 1.0

    def __post_init__(self):
        vector3(self.transmitter_m)
        unit_vector(self.transmitter_polarization)
        if not np.isfinite(self.amplitude_reference_m) or self.amplitude_reference_m <= 0:
            raise ValueError("amplitude reference must be positive and finite")

    def transmission(self, start, end):
        amplitude = 1.0
        for panel in self.panels:
            if panel.intersects(start, end):
                if panel.opaque:
                    return 0.0
                amplitude *= 10 ** (-panel.attenuation_db / 20)
        return amplitude

    def paths(self, receiver_reference_m, antenna):
        """Compute exact direct and broken-ray lengths for one physical antenna."""
        transmitter = vector3(self.transmitter_m)
        receiver = vector3(receiver_reference_m) + vector3(antenna.offset_m)
        tx_axis = unit_vector(self.transmitter_polarization)
        rx_axis = unit_vector(antenna.polarization)
        direct_length = float(np.linalg.norm(receiver - transmitter))
        if direct_length <= 1e-9:
            raise ValueError("transmit and receive antennas must be distinct")
        direct_direction = (receiver - transmitter) / direct_length
        field = tx_axis - np.dot(tx_axis, direct_direction) * direct_direction
        coupling = float(np.dot(rx_axis, field))
        gain = self.transmission(transmitter, receiver) * coupling * self.amplitude_reference_m / direct_length
        paths = []
        if abs(gain) > 1e-12:
            paths.append(PropagationPath(direct_length / C0, complex(gain), "LOS"))
        for index, scatterer in enumerate(self.scatterers):
            point = vector3(scatterer.position_m)
            incident, outgoing = point - transmitter, receiver - point
            first_leg, second_leg = float(np.linalg.norm(incident)), float(np.linalg.norm(outgoing))
            if min(first_leg, second_leg) <= 1e-9:
                raise ValueError("a point scatterer cannot coincide with an antenna")
            incident, outgoing = incident / first_leg, outgoing / second_leg
            field = tx_axis - np.dot(tx_axis, incident) * incident
            field = field - np.dot(field, outgoing) * outgoing
            angle = scatterer.polarization_rotation_rad
            field = field * np.cos(angle) + np.cross(outgoing, field) * np.sin(angle)
            coupling = float(np.dot(rx_axis, field))
            length = first_leg + second_leg
            transmission = self.transmission(transmitter, point) * self.transmission(point, receiver)
            gain = scatterer.coefficient * transmission * coupling * self.amplitude_reference_m / length
            if abs(gain) > 1e-12:
                paths.append(PropagationPath(length / C0, complex(gain), f"scatterer_{index}"))
        return paths

    def antenna_paths(self, receiver_reference_m, antennas):
        if not antennas:
            raise ValueError("at least one receive antenna is required")
        return tuple(self.paths(receiver_reference_m, antenna) for antenna in antennas)
