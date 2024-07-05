from thercy.constants import PartType
from thercy.state import StatePoint

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

    def solve(self, inlets: dict[str, StatePoint]):
        outlets = {}

        inlet_label, inlet_state = next(iter(inlets.items()))
        outlet_state = inlet_state.clone()

        outlet_state['Q'] = 1.0
        outlet_state['P'] = inlet_state['P']
        outlet_state.properties('Q', 'P')

        self._deltaH = outlet_state['H'] - inlet_state['H']

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.clone()

        return outlets
