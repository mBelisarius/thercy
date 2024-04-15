from thercy.constants import PartType
from thercy.state import StatePoint

from .base_part import BasePart, Connection


class Turbine(BasePart):
    def __init__(self, label, p_out, eta=1.0, connections=None):
        """
        Parameters
        ----------
        label : str
        p_out : float
        eta : float
        connections : list[Connection]

        """
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

    def solve(self, inlets: dict[str, StatePoint]):
        outlets = {}

        inlet_label, inlet_state = next(iter(inlets.items()))
        outlet_state = inlet_state.clone()

        outlet_state['P'] = self._p_out
        outlet_state['S'] = inlet_state['S']
        outlet_state.properties('P', 'S')

        outlet_state['H'] = inlet_state['H'] - self._eta * (inlet_state['H'] - outlet_state['H'])
        outlet_state.properties('P', 'H')

        self._deltaH = outlet_state['H'] - inlet_state['H']

        for outlet in self.get_outlets(inlet_label):
            outlets[outlet.label] = outlet_state.clone()

        return outlets
