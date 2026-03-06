import numpy as np
from .base import LearningRule


class STDPRule(LearningRule):
    """
    Exponential pair-based STDP.

    Synapse variables:
        w      : synaptic weight
        apre   : presynaptic eligibility trace  (decays with taupre)
        apost  : postsynaptic eligibility trace (decays with taupost)

    On presynaptic spike:
        v_post += w               (weight delivery)
        apre   += Apre_delta      (increment LTP trace)
        w       = clip(w + apost) (LTD: post fired recently → decrease w)

    On postsynaptic spike:
        apost  += Apost_delta     (increment LTD trace)
        w       = clip(w + apre)  (LTP: pre fired recently → increase w)

    Config keys:
        taupre      (ms)   : LTP trace time constant
        taupost     (ms)   : LTD trace time constant
        Apre_delta         : LTP increment
        Apost_delta        : LTD increment (negative)
        wmax               : weight upper bound
        wmin               : weight lower bound
        w_init_mean        : mean of initial weight uniform distribution
        homeostasis_norm   : target L1 column norm (None = disabled)
        seed               : random seed for weight init
    """

    def get_synapse_model(self) -> str:
        return """
            w         : 1
            dapre/dt  = -apre  / taupre  : 1 (event-driven)
            dapost/dt = -apost / taupost : 1 (event-driven)
        """

    def get_on_pre(self) -> str:
        c = self.config
        wmin = c.get('wmin', 0.0)
        wmax = c.get('wmax', 0.3)
        return f"""
            v_post     += w
            apre       += Apre_delta
            w           = clip(w + apost, {wmin}, {wmax})
        """

    def get_on_post(self) -> str:
        c = self.config
        wmin = c.get('wmin', 0.0)
        wmax = c.get('wmax', 0.3)
        return f"""
            apost      += Apost_delta
            w           = clip(w + apre, {wmin}, {wmax})
        """

    def get_namespace(self) -> dict:
        from brian2 import ms
        c = self.config
        return {
            'taupre':      c.get('taupre',      20) * ms,
            'taupost':     c.get('taupost',     20) * ms,
            'Apre_delta':  c.get('Apre_delta',   0.01),
            'Apost_delta': c.get('Apost_delta', -0.0105),
        }

    def apply_homeostasis(self, weights: np.ndarray) -> np.ndarray:
        """
        L1 column normalization: keeps total input per hidden neuron constant.
        Prevents some neurons from dominating while others go silent.
        """
        target = self.config.get('homeostasis_norm', None)
        if target is None:
            return weights

        wmin = self.config.get('wmin', 0.0)
        wmax = self.config.get('wmax', 0.3)

        col_norm = weights.sum(axis=0)
        col_norm = np.where(col_norm > 0, col_norm, 1.0)
        weights = weights / col_norm[None, :] * target
        return np.clip(weights, wmin, wmax)
