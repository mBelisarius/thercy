from thercy.constants import PartType
from thercy.state import StatePoint

from .base_part import BasePart, Connection


class Trap(BasePart):
    def __init__(self, label, p_out, connections=None):
        """
        Parameters
        ----------
        label : str
        p_out : float
        connections : list[Connection]

        """
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

    def solve(self, inlets: dict[str, StatePoint]):
        outlets = {}

        inlet_label, inlet_state = next(iter(inlets.items()))
        outlet_state = inlet_state.clone()

        outlet_state['P'] = self._p_out
        outlet_state['H'] = inlet_state['H']
        outlet_state.properties('P', 'H')

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.clone()

        return outlets
