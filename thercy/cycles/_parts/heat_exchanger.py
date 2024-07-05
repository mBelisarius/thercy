from thercy.constants import PartType
from thercy.state import StatePoint

from .base_part import BasePart, Connection
from .condenser import Condenser
from .evaporator import Evaporator


class HeatExchanger(BasePart):
    def __init__(self, label, dt=0., connections=None):
        """
        Parameters
        ----------
        label : str
        dt : float
        connections : list[Connection]

        """
        super().__init__(
            label,
            PartType.REHEATER_OPEN,
            connections,
        )

        self._dt = dt
        self._deltaH = 0.0

    @property
    def deltaH(self):
        return self._deltaH

    def solve(self, inlets: dict[str, StatePoint]):
        inlet_cond: str = None
        inlet_evap: str = None
        temperatures = []

        for label, state in inlets.items():
            temperatures.append(state['T'])

        temperature_max = max(temperatures)
        for label, state in inlets.items():
            if state['T'] == temperature_max:
                inlet_cond = label
            else:
                inlet_evap = label

        outlet_state_cond = inlets[inlet_cond].clone()
        outlet_state_cond['Q'] = 0.0
        outlet_state_cond['P'] = inlets[inlet_cond]['P']
        outlet_state_cond.properties('Q', 'P')
        deltaH_cond = outlet_state_cond['H'] - inlets[inlet_cond]['H']

        outlet_state_evap = inlets[inlet_evap].clone()
        outlet_state_evap['Q'] = 1.0
        outlet_state_evap['P'] = inlets[inlet_evap]['P']
        outlet_state_evap.properties('Q', 'P')
        deltaH_evap = outlet_state_evap['H'] - inlets[inlet_evap]['H']

        # Adiabatic proccess: deltaH = 0
        # self._deltaH = deltaH_cond - deltaH_evap

        # outlet_state_evap = inlets[inlet_evap].clone()
        # if abs(deltaH_evap) > abs(deltaH_cond):
        #     outlet_state_evap['H'] = inlets[inlet_evap]['H'] - deltaH_cond
        #     outlet_state_evap['P'] = inlets[inlet_evap]['P']
        #     outlet_state_evap.properties('Q', 'P')
        # else:
        #     outlet_state_evap['H'] = inlets[inlet_evap]['H'] - deltaH_cond
        #     outlet_state_evap['Q'] = 1.0
        #     outlet_state_evap.properties('Q', 'H')

        outlets = {}
        for outlet in self.get_outlets(inlet_cond):
            outlets[outlet.label] = outlet_state_cond.clone()
        for outlet in self.get_outlets(inlet_evap):
            outlets[outlet.label] = outlet_state_evap.clone()

        return outlets
