from thercy.constants import PartType, Property
from thercy.state import StateCycle, StateGraph

from .base_part import BasePart, Connection


class Turbine(BasePart):
    """
    Turbine part where the outlet pressure is ``p_out``.

    Parameters
    ----------
    label : str
        Label for this part.
    p_out : float
        Pressure of the outlet.
    eta : float, optional
        Isentropic efficiency. Default: 1.0
    connections : list[Connection]
        List of connections for this part.
    """
    def __init__(self, label, p_out, eta=1.0, connections=None):
        super().__init__(
            label,
            PartType.TURBINE,
            connections,
        )

        self._p_out = p_out
        self._eta = eta
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
        outlet_state[Property.S.value] = inlet_state[Property.S.value]
        StateCycle.calculate_props(outlet_state, graph.fluid, 'P', 'S')
        # outlet_state[Property.Y.value] = inlet_state[Property.Y.value]

        outlet_state[Property.H.value] = inlet_state[Property.H.value] - self._eta * (inlet_state[Property.H.value] - outlet_state[Property.H.value])
        StateCycle.calculate_props(outlet_state, graph.fluid, 'P', 'H')

        self._deltaH = (outlet_state[Property.H.value] - inlet_state[Property.H.value]) * inlet_state[Property.Y.value]

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.copy()

        return outlets
