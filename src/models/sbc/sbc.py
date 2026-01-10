from brian2 import *

class SimpleBinaryClassification:
    def __init__(self):
        # -------- Input neuron (SpikeGeneratorGroup) --------
        self.input = SpikeGeneratorGroup(
            N=1,
            indices=[],          # will be set by caller
            times=[]*ms
        )

        # -------- Output neuron (LIF) --------
        eqs = '''
        dv/dt = (-v) / (3*ms) : 1
        '''

        self.output = NeuronGroup(
            N=1,
            model=eqs,
            threshold='v > 1.0',
            reset='v = 0.0',
            method='exact'
        )

        # -------- STDP Synapse --------
        self.syn = Synapses(
            self.input,
            self.output,
            model='''
            w : 1
            dapre/dt = -apre / (4*ms) : 1 (event-driven)
            dapost/dt = -apost / (4*ms) : 1 (event-driven)
            ''',
            on_pre='''
            v_post += w
            apre += 0.01
            w = clip(w + apost, 0, 1)
            ''',
            on_post='''
            apost += -0.01
            w = clip(w + apre, 0, 1)
            '''
        )

        self.syn.connect()
        self.syn.w = 1

        # -------- Monitors --------
        self.spikes_in = SpikeMonitor(self.input)
        self.spikes_out = SpikeMonitor(self.output)
        self.wmon = StateMonitor(self.syn, 'w', record=0)
        self.vmon = StateMonitor(self.output, 'v', record=0)

    def get_objects(self):
        return [
            self.input,
            self.output,
            self.syn,
            self.spikes_in,
            self.spikes_out,
            self.wmon,
            self.vmon
        ]
