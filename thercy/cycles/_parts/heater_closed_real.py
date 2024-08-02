import numpy as np

from thercy.constants import PartType, Property
from thercy.state import StateCycle, StateGraph

from .base_part import BasePart, Connection


class HeaterClosedReal(BasePart):
    def __init__(self, label, connections=None):
        """
        Parameters
        ----------
        label : str
        connections : list[Connection]

        """
        super().__init__(
            label,
            PartType.REHEATER_OPEN,
            connections,
        )

        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, graph: StateGraph, inlets: list[str]):
        inlet_hp: dict[str, np.ndarray] = {}
        inlets_lp: dict[str, np.ndarray] = {}
        pressures = []

        for label in inlets:
            state = graph.get_state((label, self.label))
            pressures.append(state[Property.P.value])

        pressure_max = np.max(pressures)
        for label in inlets:
            state = graph.get_state((label, self.label))
            if state[Property.P.value] == pressure_max:
                inlet_hp[label] = state
            else:
                inlets_lp[label] = state

        outlet_lp_state = StateCycle.new_empty_state()
        outlet_hp_state = StateCycle.new_empty_state()
        outlets = {}

        partial_y_lp = 0.0
        partial_p_lp = 0.0
        partial_h_lp = 0.0
        partial_t_lp = float('inf')
        for inlet in inlets_lp.values():
            partial_y_lp += inlet[Property.Y.value]
            partial_p_lp += inlet[Property.Y.value] * inlet[Property.P.value]
            partial_h_lp += inlet[Property.Y.value] * inlet[Property.H.value]
            partial_t_lp = min(partial_t_lp, inlet[Property.T.value])

        outlet_lp_state[Property.P.value] = partial_p_lp / partial_y_lp
        outlet_lp_state[Property.Q.value] = 0.0
        StateCycle.calculate_props(outlet_lp_state, graph.fluid, 'P', 'Q')
        outlet_lp_state[Property.Y.value] = partial_y_lp
        # outlet_lp_state['Y'] = partial_y_lp

        dH = partial_y_lp * (outlet_lp_state[Property.H.value] - partial_h_lp)

        partial_y_hp = 0.0
        partial_p_hp = 0.0
        partial_h_hp = 0.0
        for inlet in inlet_hp.values():
            partial_y_hp += inlet[Property.Y.value]
            partial_p_hp += inlet[Property.Y.value] * inlet[Property.P.value]
            partial_h_hp += inlet[Property.Y.value] * inlet[Property.H.value]

        outlet_hp_state[Property.P.value] = partial_p_hp / partial_y_hp
        outlet_hp_state[Property.H.value] = partial_h_hp + dH / partial_y_hp
        StateCycle.calculate_props(outlet_hp_state, graph.fluid, 'P', 'H')
        outlet_hp_state[Property.Y.value] = partial_y_hp
        # outlet_hp_state['Y'] = partial_y_hp

        for outlet in self.get_outlets(next(iter(inlet_hp.keys()))):
            outlets[outlet.label] = outlet_hp_state.copy()

        for outlet in self.get_outlets(next(iter(inlets_lp.keys()))):
            outlets[outlet.label] = outlet_lp_state.copy()

        return outlets
