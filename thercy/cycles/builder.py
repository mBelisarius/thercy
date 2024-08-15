from thercy.state import StateGraph
from thercy.utils import list_like

from .cycle import Cycle
from ._parts import *


class CycleBuilder:
    """
    Build a thermodynamic `Cycle`. It contains methods for adding various parts
    to the cycle and connecting them together.

    Parameters
    ----------
    fluid : str
        Fluid to be used in the cycle.
    """

    def __init__(self, fluid):
        self._connections: dict[str: list[(list[str], list[str])]] = {}
        self._fluid = fluid
        self._graph = StateGraph(fluid)
        self._parts: dict[str: BasePart] = {}

    def build(self, fraction_base=1000.0):
        """
        Builds and returns a `Cycle` object.

        Parameters
        ----------
        fraction_base : float, optional
            The maximum value for fractions to be used for the cycle.
            If 1.0, then fractions will be a percentage. Default: 1000.0

        Returns
        -------
        Cycle
            The `Cycle` object created using the given inputs.
        """
        for label, part in self._parts.items():
            connections = []
            for conn in self._connections[label]:
                inlets = [self._parts[inlet] for inlet in conn[0]]
                outlets = [self._parts[outlet] for outlet in conn[1]]
                connections.append(Connection(inlets, outlets))

            part.connections = connections
            self._graph.add_part(part)

        cycle = Cycle(self._fluid, self._graph, fraction_base)
        self._graph = StateGraph(self._fluid)
        self._parts = {}

        return cycle

    def add_condenser(self, label, inlet, outlet):
        """
        Adds a `Condenser` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet : str
            The label of the inlet for this part.
        outlet : str
            The label of the outlet for this part.

        Returns
        -------
        self
            The current instance of the builder.
        """
        condensator = Condenser(label)
        self._parts[label] = condensator
        self._connections[label] = [([inlet], [outlet])]
        return self

    def add_evaporator(self, label, inlet, outlet):
        """
        Adds a `Evaporator` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet : str
            The label of the inlet for this part.
        outlet : str
            The label of the outlet for this part.

        Returns
        -------
        self
            The current instance of the builder.
        """
        evaporator = Evaporator(label)
        self._parts[label] = evaporator
        self._connections[label] = [([inlet], [outlet])]
        return self

    def add_heat_exchanger(self, label, inlet_lt, inlet_ht, outlet_lt, outlet_ht, dt=0.0):
        """
        Adds a `HeatExchanger` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet_lt : str
            The label of the low temperature inlet for this part.
        inlet_ht : str
            The label of the high temperature inlet for this part.
        outlet_lt : str
            The label of the low temperature outlet for this part.
        outlet_ht : str
            The label of the high temperature outlet for this part.
        dt : float, optional
            Default: 0.0

        Returns
        -------
        self
            The current instance of the builder.
        """
        # TODO: What is dt?
        heat_exchanger = HeatExchanger(label, dt)
        self._parts[label] = heat_exchanger
        self._connections[label] = [([inlet_lt], [outlet_lt]), ([inlet_ht], [outlet_ht])]
        return self

    def add_heater_closed(self, label, inlets_lp, inlet_hp, outlet_lp, outlet_hp, t_out):
        """
        Adds a `HeaterClosed` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlets_lp : list[str]
            The labels of the low pressure inlets for this part.
        inlet_hp : str
            The label of the high pressure inlet for this part.
        outlet_lp : str
            The label of the low pressure outlet for this part.
        outlet_hp : str
            The label of the high pressure outlet for this part.
        t_out : float
            Temperature of the heated outlet.

        Returns
        -------
        self
            The current instance of the builder.
        """
        reheater_closed = HeaterClosed(label, t_out)
        self._parts[label] = reheater_closed
        self._connections[label] = [(inlets_lp, [outlet_lp]), ([inlet_hp], [outlet_hp])]
        return self

    def add_heater_closed_real(self, label, inlets_lp, inlet_hp, outlet_lp, outlet_hp):
        """
        Adds a `HeaterClosedReal` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlets_lp : list[str]
            The labels of the low pressure inlets for this part.
        inlet_hp : str
            The label of the high pressure inlet for this part.
        outlet_lp : str
            The label of the low pressure outlet for this part.
        outlet_hp : str
            The label of the high pressure outlet for this part.

        Returns
        -------
        self
            The current instance of the builder.
        """
        reheater_closed_real = HeaterClosedReal(label)
        self._parts[label] = reheater_closed_real
        self._connections[label] = [(inlets_lp, [outlet_lp]), ([inlet_hp], [outlet_hp])]
        return self

    def add_heater_open(self, label, inlets, outlet):
        """
        Adds a `HeaterOpen` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlets : list[str]
            The labels of the inlets for this part.
        outlet : str
            The label of the outlet for this part.

        Returns
        -------
        self
            The current instance of the builder.
        """
        reheater_open = HeaterOpen(label)
        self._parts[label] = reheater_open
        self._connections[label] = [(inlets, [outlet])]
        return self

    def add_pump(self, label, inlet, outlet, p_out, eta=1.0):
        """
        Adds a `Pump` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet : str
            The label of the inlet for this part.
        outlet : str
            The label of the outlet for this part.
        p_out : float
            Pressure of the outlet.
        eta : float, optional
            Isentropic efficiency. Default: 1.0

        Returns
        -------
        self
            The current instance of the builder.
        """
        pump = Pump(label, p_out, eta=eta)
        self._parts[label] = pump
        self._connections[label] = [([inlet], [outlet])]
        return self

    def add_steam_generator(self, label, inlet, outlet, prop, value):
        """
        Adds a `SteamGenerator` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet : str
            The label of the inlet for this part.
        outlet : str
            The label of the outlet for this part.
        prop : str
            Known property at the outlet.
        value : float
            Value of the known property.

        Returns
        -------
        self
            The current instance of the builder.
        """
        heat_source = SteamGenerator(label, prop, value)
        self._parts[label] = heat_source
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_trap(self, label, inlet, outlet, p_out):
        """
        Adds a `Trap` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet : str
            The label of the inlet for this part.
        outlet : str
            The label of the outlet for this part.
        p_out : float
            Pressure of the outlet.

        Returns
        -------
        self
            The current instance of the builder.
        """
        trap = Trap(label, p_out)
        self._parts[label] = trap
        self._connections[label] = [([inlet], [outlet])]

        return self

    def add_turbine(self, label, inlet, outlets, p_out, eta=1.0):
        """
        Adds a `Turbine` to the system.

        Parameters
        ----------
        label : str
            The label for this part.
        inlet : str
            The label of the inlet for this part.
        outlets : list[str] | str
            The labels of the outlets for this part.
        p_out : float
            Pressure of the outlet.
        eta : float, optional
            Isentropic efficiency. Default: 1.0

        Returns
        -------
        self
            The current instance of the builder.
        """
        turbine = Turbine(label, p_out, eta=eta)
        self._parts[label] = turbine
        _outlets = outlets if list_like(outlets) else [outlets]
        self._connections[label] = [([inlet], _outlets)]
        return self
