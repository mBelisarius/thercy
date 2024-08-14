from thercy.state import StateGraph
from thercy.utils import list_like

from .cycle import Cycle
from ._parts import *


class CycleBuilder:
    _fluid: str
    _graph: StateGraph
    _parts: dict[str: BasePart]

    def __init__(self, fluid: str):
        self._connections: dict[str: list[(list[str], list[str])]] = {}
        self._fluid = fluid
        self._graph = StateGraph(fluid)
        self._parts: dict[str: BasePart] = {}

    def build(self, fraction_base=1000.0):
        for label, part in self._parts.items():
            connections = []
            for conn in self._connections[label]:
                inlets = [self._parts[inlet] for inlet in conn[0]]
                outlets = [self._parts[outlet] for outlet in conn[1]]
                connections.append(Connection(inlets, outlets))

            part.connections = connections
            self._graph.add_part(part)

        return Cycle(self._fluid, self._graph, fraction_base)

    def add_condenser(self, label: str, inlet: str, outlet: str):
        condensator = Condenser(label)
        self._parts[label] = condensator
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_evaporator(self, label: str, inlet: str, outlet: str):
        evaporator = Evaporator(label)
        self._parts[label] = evaporator
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_heat_exchanger(self, label: str, inlet_lt: str, inlet_ht: str, outlet_lt: str, outlet_ht: str, dt: float = 0.):
        heat_exchanger = HeatExchanger(label, dt)
        self._parts[label] = heat_exchanger
        self._connections[label] = [([inlet_lt], [outlet_lt]), ([inlet_ht], [outlet_ht])]

        return self

    def add_heater_closed(self, label: str, inlets_lp: list[str], inlet_hp: str, outlet_lp: str, outlet_hp: str, t_out: float):
        reheater_closed = HeaterClosed(label, t_out)
        self._parts[label] = reheater_closed
        self._connections[label] = [(inlets_lp, [outlet_lp]), ([inlet_hp], [outlet_hp])]

        return self

    def add_heater_closed_real(self, label: str, inlets_lp: list[str], inlet_hp: str, outlet_lp: str, outlet_hp: str):
        reheater_open = HeaterClosedReal(label)
        self._parts[label] = reheater_open
        self._connections[label] = [(inlets_lp, [outlet_lp]), ([inlet_hp], [outlet_hp])]

        return self

    def add_heater_open(self, label: str, inlets: list[str], outlet: str):
        reheater_open = HeaterOpen(label)
        self._parts[label] = reheater_open
        self._connections[label] = [(inlets, [outlet])]

        return self

    def add_pump(self, label: str, inlet: str, outlet: str, p_out: float, eta: float = 1.0):
        pump = Pump(label, p_out, eta=eta)
        self._parts[label] = pump
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_steam_generator(self, label: str, inlet: str, outlet: str, prop: str, value: float):
        heat_source = SteamGenerator(label, prop, value)
        self._parts[label] = heat_source
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_trap(self, label, inlet, outlet, p_out):
        """
        Parameters
        ----------
        label : str
        inlet : str
        outlet : str
        p_out : float

        """
        trap = Trap(label, p_out)
        self._parts[label] = trap
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_turbine(self, label, inlet, outlets, p_out, eta=1.0):
        """
        Parameters
        ----------
        label : str
        inlet : str
        outlets : str | list[str]
        p_out : float
        eta : float

        """
        turbine = Turbine(label, p_out, eta=eta)
        self._parts[label] = turbine

        _outlets = outlets if list_like(outlets) else [outlets]
        self._connections[label] = [([inlet], _outlets)]

        return self
