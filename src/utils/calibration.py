"""Measured auto-calibration of input current to a target firing rate.

The input layer of these SNNs is a group of *independent* LIF neurons (no
recurrent or lateral connections), so the mean firing rate produced by a current
matrix can be measured exactly with a single throwaway Brian2 run. This module
finds a scalar gain ``s`` such that ``s * I`` drives the input layer at an
approximately constant firing **rate** (Hz) regardless of the audio file or its
length — which is what makes a hand-tuned STDP setup transfer across utterances
(and from Vox1 to Vox2).
"""

import numpy as np
from brian2 import (
    NeuronGroup, SpikeMonitor, Network, TimedArray, ms, second, defaultclock,
)


# Default input-neuron model — matches the adaptive-LIF input layer used in the
# training scripts. Override via `neuron_params` if a script diverges.
DEFAULT_INPUT_NEURON_PARAMS = {
    "tau_m":       40 * ms,
    "tau_a":      100 * ms,
    "tau_current":  1 * ms,
    "beta":         1.0,
    "v_th_in":      1.0,
    "refractory":   2 * ms,
}

_EQS_IN = """
dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
da/dt = -a / tau_a : 1
"""


def _measure_rate(I, s, dt, neuron_params):
    """Mean per-neuron firing rate (Hz) of the input layer driven by ``s * I``."""
    N_in, T = I.shape
    duration_s = T * float(dt / second)

    ns = dict(neuron_params)
    G = NeuronGroup(
        N_in, _EQS_IN,
        threshold="v > v_th_in",
        reset="v=0; a+=beta",
        refractory=ns["refractory"],
        method="euler",
        namespace={k: ns[k] for k in ("tau_m", "tau_a", "tau_current",
                                      "beta", "v_th_in")},
    )
    G.namespace["I_timed"] = TimedArray((s * I).T.astype(float), dt=dt)

    mon = SpikeMonitor(G)
    net = Network(G, mon)
    net.run(T * dt)

    n_spikes = int(mon.num_spikes)
    return n_spikes / (N_in * duration_s + 1e-12)


def calibrate_current_to_rate(
    I,
    *,
    dt=1 * ms,
    target_rate_hz=10.0,
    neuron_params=None,
    tol=0.05,
    max_iter=12,
    s_init=1.0,
    verbose=False,
):
    """Scale ``I`` so the input layer fires at ~``target_rate_hz`` (mean over neurons).

    The firing rate is monotonically non-decreasing in the gain ``s``, so we first
    expand a bracket ``[s_lo, s_hi]`` around the target then bisect it.

    Parameters
    ----------
    I : np.ndarray (N_in, T)
        Encoded input current (e.g. from ``compute_spike_input_current``).
    dt : Brian2 time quantity
        Simulation timestep; must match the training ``defaultclock.dt``.
    target_rate_hz : float
        Desired mean per-neuron input firing rate.
    neuron_params : dict or None
        Input-neuron constants; defaults to ``DEFAULT_INPUT_NEURON_PARAMS``.
    tol : float
        Relative tolerance on the achieved rate.
    max_iter : int
        Max bisection iterations after bracketing.
    s_init : float
        Initial gain guess used to seed the bracket search.

    Returns
    -------
    I_scaled : np.ndarray (N_in, T)
        ``s * I``.
    s : float
        The chosen gain.
    measured_rate_hz : float
        Rate achieved by ``s``.
    """
    if neuron_params is None:
        neuron_params = DEFAULT_INPUT_NEURON_PARAMS

    defaultclock.dt = dt  # keep the throwaway run consistent with training

    def rate(s):
        return _measure_rate(I, s, dt, neuron_params)

    # ── Bracket: find s_lo (rate <= target) and s_hi (rate >= target) ────────
    s = float(s_init)
    r = rate(s)
    if verbose:
        print(f"    [calib] s={s:.4g} -> {r:.2f} Hz (target {target_rate_hz})")

    if abs(r - target_rate_hz) <= tol * target_rate_hz:
        return s * I, s, r

    if r < target_rate_hz:
        s_lo, r_lo = s, r
        s_hi = s
        for _ in range(40):
            s_hi *= 2.0
            r_hi = rate(s_hi)
            if verbose:
                print(f"    [calib] expand hi s={s_hi:.4g} -> {r_hi:.2f} Hz")
            if r_hi >= target_rate_hz:
                break
        else:
            # Could not reach target even at very high gain — return best effort.
            return s_hi * I, s_hi, r_hi
    else:
        s_hi, r_hi = s, r
        s_lo = s
        for _ in range(40):
            s_lo *= 0.5
            r_lo = rate(s_lo)
            if verbose:
                print(f"    [calib] expand lo s={s_lo:.4g} -> {r_lo:.2f} Hz")
            if r_lo <= target_rate_hz:
                break
        else:
            return s_lo * I, s_lo, r_lo

    # ── Bisect ───────────────────────────────────────────────────────────────
    s_mid, r_mid = s_lo, r_lo
    for _ in range(max_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        r_mid = rate(s_mid)
        if verbose:
            print(f"    [calib] bisect s={s_mid:.4g} -> {r_mid:.2f} Hz")
        if abs(r_mid - target_rate_hz) <= tol * target_rate_hz:
            break
        if r_mid < target_rate_hz:
            s_lo = s_mid
        else:
            s_hi = s_mid

    return s_mid * I, s_mid, r_mid
