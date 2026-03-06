from .base import Layer
from typing import Any, Dict


class AdaptiveLIF(Layer):
    """
    Adaptive LIF neuron — used for the auditory input encoding layer.

    Equations:
        dv/dt = (-v + I_input(t, i) - a) / tau_m
        da/dt = -a / tau_a

    On spike: v = 0; a += beta
    Input current injected via Brian2 TimedArray (I_input).
    """

    def get_equations(self) -> str:
        return """
            dv/dt = (-v + I_input(t, i) - a) / tau_m : 1
            da/dt = -a / tau_a : 1
        """

    def get_threshold(self) -> str:
        return f"v > {self.config.get('v_th', 1.0)}"

    def get_reset(self) -> str:
        return "v = 0; a += beta"

    def get_namespace(self, dt_ms: float) -> Dict[str, Any]:
        from brian2 import ms
        p = self.config
        return {
            'tau_m': p.get('tau_m', 10) * ms,
            'tau_a': p.get('tau_a', 100) * ms,
            'beta':  p.get('beta', 0.2),
        }
        # Note: I_input TimedArray is injected by SNNNetwork.run_simulation()
        # and added to the namespace separately.

    def get_initial_state(self) -> Dict[str, Any]:
        return {'v': 0, 'a': 0}


class SimpleLIF(Layer):
    """
    Standard LIF neuron — used for hidden/output layers.

    Equations:
        dv/dt = -v / tau_m

    On spike: v = 0
    """

    def get_equations(self) -> str:
        return """
            dv/dt = -v / tau_m : 1
        """

    def get_threshold(self) -> str:
        return f"v > {self.config.get('v_th', 1.0)}"

    def get_reset(self) -> str:
        return "v = 0"

    def get_namespace(self, dt_ms: float) -> Dict[str, Any]:
        from brian2 import ms
        p = self.config
        return {
            'tau_m': p.get('tau_m', 10) * ms,
        }

    def get_initial_state(self) -> Dict[str, Any]:
        return {'v': 0}
