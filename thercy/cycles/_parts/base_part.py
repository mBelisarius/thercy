from abc import ABC, abstractmethod

from thercy.constants import PartType
from thercy.state import StateGraph, StatePoint


class Connection:
    def __init__(self, inlets, outlets):
        """
        Parameters
        ----------
        inlets : list[BasePart]
        outlets : list[BasePart]

        """
        self._inlets = inlets
        self._outlets = outlets

    @property
    def inlets(self):
        return self._inlets

    @property
    def outlets(self):
        return self._outlets


class BasePart(ABC):
    def __init__(self, label, type, connections=None):
        """
        Parameters
        ----------
        label : str
        type : PartType
        connections : list[Connection]

        """
        if connections is None:
            connections = []

        self._label: str = label
        self._type: PartType = type
        self._internal_conn: list[Connection] = connections

    @property
    def label(self):
        return self._label

    @property
    def type(self):
        return self._type

    @property
    def connections(self):
        return self._internal_conn

    @connections.setter
    def connections(self, connections):
        """
        Parameters
        ----------
        connections : list[Connection]

        """
        for con in connections:
            for inlet in con.inlets + con.outlets:
                if not isinstance(inlet, BasePart):
                    raise ValueError("part must be a BasePart")

        self._internal_conn = connections

    @property
    def inlet_parts(self):
        inlets = []
        for conn in self._internal_conn:
            for inlet in conn.inlets:
                inlets.append(inlet)

        return inlets

    @property
    def outlet_parts(self):
        outlets = []
        for conn in self._internal_conn:
            for outlet in conn.outlets:
                outlets.append(outlet)

        return outlets

    @property
    @abstractmethod
    def deltaH(self):
        raise NotImplementedError()

    def get_inlets(self, outlet: str):
        for conn in self._internal_conn:
            for outl in conn.outlets:
                if outl.label == outlet:
                    return conn.inlets

        return None

    def get_outlets(self, inlet: str):
        for conn in self._internal_conn:
            for inl in conn.inlets:
                if inl.label == inlet:
                    return conn.outlets

        return None

    def solve_conserv(self, inlets_state: dict[str, StatePoint], outlets_state: dict[str, StatePoint]):
        residual = [0.0, 0.0]

        for conn in self.connections:
            y_inl = 0.0
            h_inl = 0.0
            for label, state in inlets_state.items():
                if label in [p.label for p in conn.inlets]:
                    y_inl += state['Y']
                    h_inl += state['Y'] * state['H']

            y_outl = 0.0
            h_outl = 0.0
            for label, state in outlets_state.items():
                if label in [p.label for p in conn.outlets]:
                    y_outl += state['Y']
                    h_outl += state['Y'] * state['H']

            y = (y_inl + y_outl) / 2.0
            residual[0] += abs(y_outl - y_inl)
            residual[1] += h_outl - h_inl - y * self.deltaH

        return residual

    @abstractmethod
    def solve(self, inlets: dict[str, StatePoint]):
        raise NotImplementedError()
