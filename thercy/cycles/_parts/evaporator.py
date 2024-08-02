from thercy.constants import PartType, Property
from thercy.state import StateCycle, StateGraph

from .base_part import BasePart, Connection


class Evaporator(BasePart):
    def __init__(self, label, connections=None):
        """
        Parameters
        ----------
        label : str
        connections : list[Connection]

        """
        super().__init__(
            label,
            PartType.EVAPORATOR,
            connections,
        )

        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, graph: StateGraph, inlets: list[str]):
        outlets = {}

        inlet_label = inlets[0]
        inlet_state = graph.get_state((inlet_label, self.label))

        outlet_state = StateCycle.new_empty_state()
        outlet_state[Property.Q.value] = 1.0
        outlet_state[Property.P.value] = inlet_state[Property.P.value]
        StateCycle.calculate_props(outlet_state, graph.fluid, 'Q', 'P')
        outlet_state[Property.Y.value] = inlet_state[Property.Y.value]

        self._deltaH = outlet_state[Property.H.value] - inlet_state[Property.H.value]

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.copy()

        return outlets
