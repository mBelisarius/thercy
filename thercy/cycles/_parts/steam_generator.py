from thercy.constants import PartType, PropertyInfo
from thercy.state import StatePoint

from .base_part import BasePart, Connection


class SteamGenerator(BasePart):
    _prop: str

    def __init__(self, label, prop, value, connections=None):
        """
        Parameters
        ----------
        label : str
        prop : str
        value : float
        connections : list[Connection]

        """
        super().__init__(
            label,
            PartType.HEAT_SOURCE,
            connections,
        )

        self._prop = PropertyInfo.get_strkey(prop)
        self._value = value
        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, inlets: dict[str, StatePoint]):
        outlets = {}

        inlet_label, inlet_state = next(iter(inlets.items()))
        outlet_state = inlet_state.clone()

        outlet_state[self._prop] = self._value
        outlet_state['P'] = inlet_state['P']
        outlet_state.properties(self._prop, 'P')

        self._deltaH = outlet_state['H'] - inlet_state['H']

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.clone()

        return outlets
