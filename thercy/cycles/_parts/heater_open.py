from thercy.constants import PartType
from thercy.state import StatePoint

from .base_part import BasePart, Connection


class HeaterOpen(BasePart):
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

    def solve(self, inlets: dict[str, StatePoint]):
        outlets = {}
        outlet_state = StatePoint(next(iter(inlets.values())).fluid)

        partial_p = 0.0
        partial_y = 0.0
        for inlet in inlets.values():
            partial_p += inlet['Y'] * inlet['P']
            partial_y += inlet['Y']

        outlet_state['P'] = partial_p / partial_y
        outlet_state['Q'] = 0.0
        outlet_state.properties('P', 'Q')

        # outlet_state['Y'] = partial_y

        # Only one outlet
        for outlet in self.get_outlets(next(iter(inlets.keys()))):
            outlets[outlet.label] = outlet_state.clone()

        return outlets
