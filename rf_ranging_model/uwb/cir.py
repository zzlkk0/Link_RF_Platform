"""Convert uniformly sampled CFR measurements to a baseband CIR."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FrameCombining = Literal["coherent", "phase_aligned"]


def combine_frequency_response_frames(
    response_frames: ArrayLike,
    *,
    frame_axis: int = 0,
    combining: FrameCombining = "phase_aligned",
) -> tuple[NDArray[np.complex128], float]:
    """Combine repeated CFR measurements and return phase coherence.

    ``phase_aligned`` estimates and removes one *constant* common phase per
    frame before coherent averaging.  It deliberately does not estimate a
    frequency slope: clock error, group delay, motion, or a changing channel
    must not be silently calibrated away.  Consequently this combiner is
    appropriate for repeated measurements of a quasi-static channel, while a
    low returned coherence is evidence that the combined CIR should not be
    treated as a high-confidence observation.

    A one-dimensional input is accepted as a one-frame compatibility path.
    For a two-dimensional input, ``frame_axis`` selects the frame dimension.
    """

    frames = np.asarray(response_frames, dtype=np.complex128)
    if frames.ndim == 1:
        frames = frames[np.newaxis, :]
    elif frames.ndim == 2:
        if isinstance(frame_axis, bool) or frame_axis not in (0, 1, -1, -2):
            raise ValueError("frame_axis must select one axis of a 2-D array")
        frames = np.moveaxis(frames, frame_axis, 0)
    else:
        raise ValueError("response_frames must be a 1-D or 2-D array")
    if frames.shape[0] < 1 or frames.shape[1] < 4:
        raise ValueError("at least one frame with four frequency samples is required")
    if not np.all(np.isfinite(frames.real) & np.isfinite(frames.imag)):
        raise ValueError("response_frames must contain only finite values")
    if combining not in {"coherent", "phase_aligned"}:
        raise ValueError("combining must be 'coherent' or 'phase_aligned'")

    aligned = frames.copy()
    if combining == "phase_aligned" and frames.shape[0] > 1:
        # Start from the highest-energy frame, then refine once using the
        # first aligned average.  The scalar inner product estimates only a
        # common phase and therefore preserves all delay information.
        energies = np.sum(np.square(np.abs(frames)), axis=1)
        reference = frames[int(np.argmax(energies))]
        for _ in range(2):
            correlations = np.asarray(
                [np.vdot(reference, frame) for frame in frames],
                dtype=np.complex128,
            )
            correlation_magnitudes = np.abs(correlations)
            if np.any(correlation_magnitudes <= np.finfo(float).tiny):
                raise ValueError(
                    "a frame has no coherent overlap with the phase reference"
                )
            corrections = np.exp(-1j * np.angle(correlations))
            aligned = frames * corrections[:, np.newaxis]
            reference = np.mean(aligned, axis=0)

    combined = np.mean(aligned, axis=0)
    denominator = float(np.sum(np.linalg.norm(aligned, axis=1)))
    coherence = (
        float(frames.shape[0] * np.linalg.norm(combined) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return (
        np.asarray(combined, dtype=np.complex128),
        float(np.clip(coherence, 0.0, 1.0)),
    )


def frequency_response_to_cir(
    frequencies_hz: ArrayLike,
    response: ArrayLike,
    *,
    zero_pad_factor: int = 8,
    window: str | None = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return delay bins and complex CIR from uniformly spaced H(f).

    Zero padding interpolates the delay grid but does not create bandwidth.
    The unambiguous delay span remains 1/df.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float)
    channel = np.asarray(response, dtype=np.complex128)
    if frequencies.ndim != 1 or channel.ndim != 1 or frequencies.shape != channel.shape:
        raise ValueError("frequencies_hz and response must be matching 1-D arrays")
    if frequencies.size < 4:
        raise ValueError("at least four frequency samples are required")
    if zero_pad_factor < 1:
        raise ValueError("zero_pad_factor must be at least one")
    spacing = np.diff(frequencies)
    delta_f = float(np.mean(spacing))
    if delta_f <= 0.0 or not np.allclose(spacing, delta_f, rtol=1.0e-7, atol=1.0e-3):
        raise ValueError("frequencies_hz must be uniformly spaced and increasing")

    if window is None:
        weights = np.ones(frequencies.size)
    elif window.lower() == "hann":
        weights = np.hanning(frequencies.size)
    elif window.lower() == "hamming":
        weights = np.hamming(frequencies.size)
    else:
        raise ValueError("window must be None, 'hann', or 'hamming'")
    coherent_gain = float(np.mean(weights))
    nfft = int(frequencies.size * zero_pad_factor)
    cir = np.fft.ifft(channel * weights, n=nfft)
    if coherent_gain > 0.0:
        cir = cir / coherent_gain
    delay_s = np.arange(nfft, dtype=float) / (nfft * delta_f)
    return delay_s, np.asarray(cir, dtype=np.complex128)


def frequency_response_frames_to_cir(
    frequencies_hz: ArrayLike,
    response_frames: ArrayLike,
    *,
    frame_axis: int = 0,
    combining: FrameCombining = "phase_aligned",
    zero_pad_factor: int = 8,
    window: str | None = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.complex128], float]:
    """Combine repeated CFR frames and convert them to one CIR.

    The third return value is the normalized coherent-vector ratio in
    ``[0, 1]``.  Callers can use it as an observation-quality input; it is not
    a ranging calibration and does not by itself make a path available.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float)
    combined, coherence = combine_frequency_response_frames(
        response_frames,
        frame_axis=frame_axis,
        combining=combining,
    )
    if frequencies.ndim != 1 or frequencies.shape != combined.shape:
        raise ValueError(
            "frequencies_hz must match the frequency-sample axis of response_frames"
        )
    delays, cir = frequency_response_to_cir(
        frequencies,
        combined,
        zero_pad_factor=zero_pad_factor,
        window=window,
    )
    if not math.isfinite(coherence):  # Defensive guard for public API output.
        raise ValueError("frame coherence is not finite")
    return delays, cir, coherence
