import numpy as np
from scipy.optimize import minimize, root

from thercy.constants import PartType, Property, PropertyInfo
from thercy.state import StateGraph
from thercy.utils import norm_l1, norm_l2, norm_lmax, norm_lp


class Cycle:
    def __init__(self, fluid: str, parts: StateGraph):
        self._graph: StateGraph = parts
        self._fluid: str = fluid
        self._heat_input: float = 0.0
        self._heat_output: float = 0.0
        self._work_pumps: float = 0.0
        self._work_turbines: float = 0.0

    def __len__(self):
        return len(self._graph)

    def __str__(self):
        return str(self._graph)

    @property
    def graph(self):
        return self._graph

    @property
    def states(self):
        return self._graph.states

    def _equation_thermo(self, x: np.ndarray):
        len_props = len(Property)
        len_states = self._graph.points
        residual = np.zeros_like(x)

        for index in range(len_states):
            index_begin = index * len_props
            index_end = index_begin + len_props
            self._graph.states[index].from_array(x[index_begin:index_end])

        for part in self._graph.nodes.values():
            inlets_state = {p.label: self._graph.get_state((p.label, part.label)) for p in part.inlet_parts}
            sol = self._graph[part.label].solve(inlets_state)

            for label_outlet, value in sol.items():
                edge = (part.label, label_outlet)
                edge_index = self._graph.get_edge_index(edge)
                for prop in Property:
                    if value[prop.value] is not None:
                        self._graph.states[edge_index][prop.value] = value[prop.value]

        for index in range(len_states):
            index_begin = index * len_props
            index_end = index_begin + len_props
            residual[index_begin:index_end - 1] = (x[index_begin:index_end - 1]
                                                   - self._graph.states[index].to_array(['Y']))

        return residual

    def _iterate_thermo(self, x0: np.ndarray, xtol=1e-4, maxfev=10, verbose=0):
        sol = root(
            self._equation_thermo,
            x0,
            method='df-sane',
            options={'fatol': xtol, 'maxfev': maxfev}
        )

        len_props = len(Property)
        len_states = self._graph.points
        for index in range(len_states):
            index_begin = index * len_props
            index_end = index_begin + len_props
            self._graph.states[index].from_array(sol.x[index_begin:index_end])

        if verbose >= 3:
            print(f"{'Rankine._iterate_thermo : ':40s}"
                  f"L2(fun) = {norm_l2(sol.fun, rescale=True)}, nfev={sol.nfev}")

        return sol.x

    def _equation_conserv(self, y: np.ndarray):
        residual = np.zeros(2 * len(self._graph))

        for index in range(self._graph.points):
            self._graph.states[index]['Y'] = y[index]

        for i, part in enumerate(self._graph.nodes.values()):
            inlets_state = {p.label: self._graph.get_state((p.label, part.label)) for p in part.inlet_parts}
            outlets_state = {p.label: self._graph.get_state((part.label, p.label)) for p in part.outlet_parts}
            residual[2 * i:2 * i + 2] = self._graph[part.label].solve_conserv(inlets_state, outlets_state)

        return residual

    def _iterate(self, x_states: np.ndarray, xtol=1e-4, verbose=0):
        y = x_states[len(Property) - 1::len(Property)]

        sol_thermo = self._iterate_thermo(x_states, xtol=xtol, verbose=verbose)
        residual = self._equation_conserv(y)

        residual_mass = norm_l2(residual[0::2], rescale=True)
        residual_energy = norm_l2(residual[1::2], rescale=True)

        if verbose >= 3:
            print(f"{'Rankine._iterate_conserv : ':40s}"
                  f"{residual_energy:3e} | {residual_mass:.3e}")

        ressum = residual_energy + 1.0e6 * residual_mass
        return ressum

    def solve(self, x0, knowns=None, tol=1e-4, verbose=0):
        if knowns is None:
            raise ValueError('At least one property must be known for every state')

        len_props = len(Property)
        len_props_input = len(knowns)
        len_exclude = len_props - len_props_input
        x0_rand = 2. * np.random.rand(len_props * len(x0) // len_props_input) - 1.
        x0_complete = np.array(len(x0) * [273.15, 1e5, 1., 1e3, 1e4, 0.5, 1000.]) * (1. + 0.1 * x0_rand)

        for i in range(0, len(x0), len_props_input):
            new_index = i + (i // len_props_input) * len_exclude
            indices = [new_index + PropertyInfo.get_intkey(k) for k in knowns]
            x0_complete[indices] = x0[i:i + len_props_input]

        if 'Y' in knowns:
            x0_fraction = x0_complete[len_props - 1::len_props]
            x0_fraction *= 1000.0 / np.max(x0_fraction)
        else:
            x0_fraction = np.full(self._graph.points, 1000.0)
            x0_complete[len_props - 1::len_props] = x0_fraction

        bounds = self._graph.points * [(1.0, 2000.0), (1.0e3, 100.0e6), (1.0e-3, 1.0e4), (1.0e3, 100.0e6),
                                       (1.0, 10.0e3), (-1.0, 1.0), (1.0, 1000.0)]
        sol_conserv = minimize(
            self._iterate,
            bounds=bounds,
            x0=x0_complete,
            args=(tol, verbose,),
            method='L-BFGS-B',
            options={'ftol': tol / 10, 'maxiter': 1000}
        )

        sol_x = sol_conserv.x
        for index in range(self._graph.points):
            index_begin = index * len_props
            index_end = index_begin + len_props
            self._graph.states[index].from_array(sol_x[index_begin:index_end])

        print(sol_conserv)
        print(self._graph.states)

        # Post-processing
        for part in self._graph.nodes.values():
            y_part = sum(self._graph.states[self._graph.get_edge_index((inlet.label, part.label))]['Y']
                         for inlet in part.inlet_parts)

            match part.type:
                case PartType.CONDENSATOR:
                    self._heat_output += y_part * part.deltaH
                case PartType.HEAT_SOURCE:
                    self._heat_input += y_part * part.deltaH
                case PartType.PUMP:
                    self._work_pumps += y_part * part.deltaH
                case PartType.TURBINE:
                    self._work_turbines += y_part * part.deltaH

        return self._graph.states

    @property
    def bwr(self):
        return self._work_pumps / -self._work_turbines

    @property
    def cycle(self):
        return self._cycle

    @property
    def efficiency(self):
        return self.work / self._heat_input

    @property
    def heat_input(self):
        return self._heat_input

    @property
    def heat_output(self):
        return -self._heat_output

    def massflow(self, power):
        return power / self.work

    @property
    def work(self):
        return -(self._work_pumps + self._work_turbines)
