from thercy.constants import PartType, Property
from thercy.state import StateCycle, StateGraph

from .base_part import BasePart, Connection


class Trap(BasePart):
    """
    Trap part where the outlet pressure is ``p_out`` with an isenthalpic process
    occurring.

    Parameters
    ----------
    label : str
        Label for this part.
    p_out : float
        Pressure of the outlet.
    connections : list[Connection]
        List of connections for this part.
    """
    def __init__(self, label, p_out, connections=None):
        super().__init__(
            label,
            PartType.CONDENSATOR,
            connections,
        )

        self._p_out = p_out
        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, graph: StateGraph, inlets: list[str]):
        outlets = {}

        inlet_label = inlets[0]
        inlet_state = graph.get_state((inlet_label, self.label))

        outlet_state = StateCycle.new_empty_state()
        outlet_state[Property.P.value] = self._p_out
        outlet_state[Property.H.value] = inlet_state[Property.H.value]
        StateCycle.calculate_props(outlet_state, graph.fluid, 'P', 'H')
        # outlet_state[Property.Y.value] = inlet_state[Property.Y.value]

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.copy()

        return outlets
