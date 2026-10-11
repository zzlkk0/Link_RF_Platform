# Complex Environment Extension

`rf_ranging_model.channel.complex_scene` extends the existing shared-geometry channel API without modifying old experiments:

- `MovingPanel(panel, velocity_m_s).at(elapsed_s)` translates an attenuating or opaque box.
- `ComplexScene(scene, moving_panels, bounce_pairs).paths(receiver_m, antenna, elapsed_s)` returns the original direct/single-bounce rays and selected two-bounce point-scattering chains.
- Every broken-ray length comes from the actual antenna/scatterer coordinates. Every leg is tested against the same instantaneous obstructions. Multiple intersecting finite losses multiply in amplitude; opaque intersections remove that ray.
- `bilateral_product(local, remote, local_mask, remote_mask)` multiplies complex endpoint measurements with the intersection of valid masks. It neither conjugates nor square-roots the channel, and preserves reciprocal multipath cross terms.

This is a Python simulation/training API extension. The existing web interface's scenario menu is unchanged. The parent training project's `src/complex_environment_data.py` supplies reproducible V7 room-layout groups and archives endpoint IQ, masks, paths, gains and antenna trajectories for replay.

## Scope

Coefficients are effective point-scattering parameters, not calibrated wall-material reflection/transmission. The model has no diffraction, specular wall-intersection solver, mutual coupling, complete Bluetooth exchange, antenna-switch transients or hardware security evidence. Obstacle movement is sampled at frame times. A reflected-only ray does not establish geometric distance; absent signal remains receiver noise.

Run the platform's standard unittest discovery. `tests/test_complex_scene.py` covers invalid geometry/time, shared antenna geometry and bilateral-mask contracts; the parent project's `tests/test_conservative_v7.py` additionally checks length sums, multiplicative obstruction, moving visibility, full-blockage replay and conservative correction behavior.
