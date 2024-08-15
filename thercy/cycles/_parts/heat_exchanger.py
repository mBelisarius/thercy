import numpy as np

from thercy.constants import PartType, Property
from thercy.state import StateCycle, StateGraph

from .base_part import BasePart, Connection
from .condenser import Condenser
from .evaporator import Evaporator


class HeatExchanger(BasePart):
    """
    Heat exchanger part where the outlet of the conderser part is saturated
    liquid and the outlet of the evaporator part is saturated vapor, both with
    the same pressure as its respectively inlet.

    Parameters
    ----------
    label : str
        Label for this part.
    dt : float, optional
        Default: 0.0
    connections : list[Connection]
        List of connections for this part.
    """
    # TODO : What is dt?
    def __init__(self, label, dt=0.0, connections=None):
        super().__init__(
            label,
            PartType.HEAT_EXCHANGER,
            connections,
        )

        self._dt = dt
        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, graph: StateGraph, inlets: list[str]):
        inlet_cond: str = None
        inlet_cond_state = None
        inlet_evap: str = None
        inlet_evap_state = None
        temperatures = []

        for label in inlets:
            state = graph.get_state((label, self.label))
            temperatures.append(state[Property.T.value])

        temperature_max = np.max(temperatures)
        for label in inlets:
            state = graph.get_state((label, self.label))
            if state[Property.T.value] == temperature_max:
                inlet_cond = label
                inlet_cond_state = state
            else:
                inlet_evap = label
                inlet_evap_state = state

        outlet_cond_state = StateCycle.new_empty_state()
        outlet_cond_state[Property.Q.value] = 0.0
        outlet_cond_state[Property.P.value] = inlet_cond_state[Property.P.value]
        StateCycle.calculate_props(outlet_cond_state, graph.fluid, 'Q', 'P')
        # outlet_cond_state[Property.Y.value] = inlet_cond_state[Property.Y.value]
        deltaH_cond = (outlet_cond_state[Property.H.value] - inlet_cond_state[Property.H.value]) * inlet_cond_state[Property.Y.value]

        outlet_evap_state = StateCycle.new_empty_state()
        outlet_evap_state[Property.Q.value] = 1.0
        outlet_evap_state[Property.P.value] = inlet_evap_state[Property.P.value]
        StateCycle.calculate_props(outlet_evap_state, graph.fluid, 'Q', 'P')
        # outlet_evap_state[Property.Y.value] = inlet_evap_state[Property.Y.value]
        deltaH_evap = (outlet_evap_state[Property.H.value] - inlet_evap_state[Property.H.value]) * inlet_evap_state[Property.Y.value]

        # Adiabatic proccess: deltaH = 0
        # self._deltaH = deltaH_cond - deltaH_evap

        # outlet_state_evap = inlets[inlet_evap].clone()
        # if abs(deltaH_evap) > abs(deltaH_cond):
        #     outlet_state_evap['H'] = inlets[inlet_evap]['H'] - deltaH_cond
        #     outlet_state_evap['P'] = inlets[inlet_evap]['P']
        #     outlet_state_evap.properties('Q', 'P')
        # else:
        #     outlet_state_evap['H'] = inlets[inlet_evap]['H'] - deltaH_cond
        #     outlet_state_evap['Q'] = 1.0
        #     outlet_state_evap.properties('Q', 'H')

        outlets = {}
        for outlet in self.get_outlets(inlet_cond):
            outlets[outlet.label] = outlet_cond_state.copy()
        for outlet in self.get_outlets(inlet_evap):
            outlets[outlet.label] = outlet_evap_state.copy()

        return outlets
