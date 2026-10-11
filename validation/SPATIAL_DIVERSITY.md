# Shared-Geometry Spatial Diversity API

`rf_ranging_model.channel.spatial` adds a deterministic geometry layer without changing the existing single-antenna pipeline or web interface.

## Components

- `SpatialScene`: one transmitter, common point scatterers, attenuating panels, transmitter polarization and amplitude reference.
- `SpatialAntenna`: a receive offset in meters and a polarization vector.
- `PointScatterer`: a fixed 3-D position, shared complex coefficient and optional polarization rotation.
- `AttenuatingPanel`: finite axis-aligned bounds, attenuation in dB, or complete opacity. Intersections are evaluated separately for every ray segment.
- `scene.antenna_paths(receiver_reference_m, antennas)`: per-antenna `PropagationPath` collections compatible with the existing frequency-response and RFIC code.

The direct delay is the actual antenna separation divided by the speed of light. A scattered delay is the sum of transmitter-to-scatterer and scatterer-to-antenna lengths divided by the speed of light. Shared-ray length differences cannot exceed antenna spacing. A completely blocked scene may return no paths: callers must generate independent receiver noise, not silently insert a direct path.

```python
from rf_ranging_model.ble import allowed_cs_frequencies_hz
from rf_ranging_model.channel.multipath import frequency_response
from rf_ranging_model.channel.spatial import (
    AttenuatingPanel,
    PointScatterer,
    SpatialAntenna,
    SpatialScene,
)

scene = SpatialScene(
    transmitter_m=(0.0, 0.0, 1.2),
    scatterers=(PointScatterer((2.5, 2.0, 1.2), 0.6 + 0.2j),),
    panels=(AttenuatingPanel((2.4, -0.1, 0.4), (2.6, 0.1, 2.0), attenuation_db=20),),
    amplitude_reference_m=5.0,
)
antennas = (SpatialAntenna((0.0, 0.0, 0.0)), SpatialAntenna((0.0, 0.06, 0.0)))
paths = scene.antenna_paths((5.0, 0.0, 1.2), antennas)
frequencies = allowed_cs_frequencies_hz()
responses = [frequency_response(frequencies, antenna_paths) for antenna_paths in paths]
```

## Validation and Scope

`tests/test_spatial_channel.py` checks exact geometry, distinct antenna CFR, identical noiseless responses at coincident antennas, path-length bounds, partial/opaque panels, panel-edge diversity and polarization behavior.

The parent training project adds paired one-antenna/eight-frame, two-antenna/eight-frame and one-antenna/sixteen-frame datasets. It checks independent noise streams, identical reused antenna-0 observations, geometric labels and deterministic full-window replay. The equal-snapshot control prevents a doubled observation budget from being confused with spatial diversity alone.

This is an effective point-scatterer model. It does not implement specular wall ray tracing, calibrated material penetration, diffraction, mutual coupling, measured antenna patterns, antenna-switch transients or complete two-way Bluetooth CS exchange. It is not proof of centimeter-level ranging behind arbitrary obstacles. The optional polarization API is not a measured polarization-diversity result.
