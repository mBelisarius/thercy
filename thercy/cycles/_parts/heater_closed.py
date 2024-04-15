from thercy.constants import PartType
from thercy.state import StatePoint

from .base_part import BasePart, Connection


class HeaterClosed(BasePart):
    def __init__(self, label, t_out, connections=None):
        """
        Parameters
        ----------
        label : str
        t_out : float
        connections : list[Connection]

        """
        super().__init__(
            label,
            PartType.REHEATER_OPEN,
            connections,
        )

        self._t_out = t_out
        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, inlets: dict[str, StatePoint]):
        inlet_hp: dict[str, StatePoint] = {}
        inlets_lp: dict[str, StatePoint] = {}
        pressures = []

        for label, state in inlets.items():
            pressures.append(state['P'])

        pressure_max = max(pressures)
        for label, state in inlets.items():
            if state['P'] == pressure_max:
                inlet_hp[label] = state
            else:
                inlets_lp[label] = state

        outlet_hp_state = StatePoint(next(iter(inlets.values())).fluid)
        outlet_lp_state = StatePoint(next(iter(inlets.values())).fluid)
        outlets = {}

        partial_y_lp = 0.0
        partial_p_lp = 0.0
        partial_h_lp = 0.0
        for inlet in inlets_lp.values():
            partial_y_lp += inlet['Y']
            partial_p_lp += inlet['Y'] * inlet['P']
            partial_h_lp += inlet['Y'] * inlet['H']

        outlet_lp_state['P'] = partial_p_lp / partial_y_lp
        outlet_lp_state['Q'] = 0.0
        outlet_lp_state.properties('P', 'Q')
        # outlet_lp_state['Y'] = partial_y_lp

        # dH = partial_y_lp * outlet_lp_state['H'] - partial_h_lp

        partial_y_hp = 0.0
        partial_p_hp = 0.0
        for inlet in inlet_hp.values():
            partial_y_hp += inlet['Y']
            partial_p_hp += inlet['Y'] * inlet['P']

        outlet_hp_state['P'] = partial_p_hp / partial_y_hp
        outlet_hp_state['T'] = self._t_out
        outlet_hp_state.properties('P', 'T')
        # outlet_hp_state['Y'] = partial_y_hp

        for outlet in self.get_outlets(next(iter(inlet_hp.keys()))):
            outlets[outlet.label] = outlet_hp_state.clone()

        for outlet in self.get_outlets(next(iter(inlets_lp.keys()))):
            outlets[outlet.label] = outlet_lp_state.clone()

        return outlets
