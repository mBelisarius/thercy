from thercy.constants import PartType, Property
from thercy.state import StateCycle, StateGraph

from .base_part import BasePart, Connection


class HeaterOpen(BasePart):
    """
    Open heater part where the outlet is saturated liquid with the equivalent
    pressure (weighted average of pressures by flow rate) of its inlets.

    Parameters
    ----------
    label : str
        Label for this part.
    connections : list[Connection]
        List of connections for this part.
    """
    def __init__(self, label, connections=None):
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
        outlets = {}

        partial_p = 0.0
        partial_y = 0.0
        for inlet in inlets:
            state = graph.get_state((inlet, self.label))
            partial_p += state[Property.Y.value] * state[Property.P.value]
            partial_y += state[Property.Y.value]

        outlet_state = StateCycle.new_empty_state()
        outlet_state[Property.P.value] = partial_p / partial_y
        outlet_state[Property.Q.value] = 0.0
        StateCycle.calculate_props(outlet_state, graph.fluid, 'P', 'Q')
        # outlet_state[Property.Y.value] = partial_y

        # Only one outlet
        for outlet in self.get_outlets(inlets[0]):
            outlets[outlet.label] = outlet_state.copy()

        return outlets
