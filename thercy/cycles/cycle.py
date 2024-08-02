import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize, root

from thercy.constants import PartType, Property, PropertyInfo
from thercy.state import StateCycle, StateGraph
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
        residual = np.zeros(x.size)

        for index in range(len_states):
            self._graph.states[index] = x[index]

        for part in self._graph.nodes.values():
            # inlets_state = {p.label: self._graph.get_state((p.label, part.label)) for p in part.inlet_parts}
            inlets = [p.label for p in part.inlet_parts]
            sol = self._graph[part.label].solve(self._graph, inlets)

            for label_outlet, value in sol.items():
                edge = (part.label, label_outlet)
                edge_index = self._graph.get_edge_index(edge)
                for prop in Property:
                    if value[prop.value] is not None:
                        self._graph.states[edge_index, prop.value] = value[prop.value]

        for index in range(len_states):
            index_begin = index * len_props
            index_end = index_begin + len_props
            residual[index_begin:index_end - 1] = (x[index] - self._graph.states[index])[:-1]

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
            self._graph.states[index] = sol.x[index]

        if verbose >= 3:
            print(f"{'Rankine._iterate_thermo : ':40s}"
                  f"L2(fun) = {norm_l2(sol.fun, rescale=True)}, nfev={sol.nfev}")

        return sol.x

    def _equation_conserv(self, y: np.ndarray):
        residual = np.zeros(2 * len(self._graph))

        for index in range(self._graph.points):
            self._graph.states[index, 'Y'] = y[index]

        for i, part in enumerate(self._graph.nodes.values()):
            inlets = [p.label for p in part.inlet_parts]
            outlets = [p.label for p in part.outlet_parts]
            residual[2 * i:2 * i + 2] = self._graph[part.label].solve_conserv(self._graph, inlets, outlets)

        return residual

    def _iterate(self, y: np.ndarray, verbose=0):
        # We already have sol_thermo == self._graph.states._data, so no need to update
        y *= 1000.0 / np.max(y)
        residual = self._equation_conserv(y)

        residual_mass = norm_l2(residual[0::2], rescale=True)
        residual_energy = norm_l2(residual[1::2], rescale=True)
        ressum = residual_energy ** 2.0 + (1.0e6 * residual_mass) ** 2.0

        if verbose >= 3:
            print(f"{'Rankine._iterate_conserv : ':40s}"
                  f"{ressum:.3e} | {residual_mass:.3e} | {residual_energy:3e}")

        return ressum

    def solve(self, x0, knowns=None, tol=1e-4, verbose=0):
        if knowns is None:
            raise ValueError('At least one property must be known for every state')

        len_knowns = len(knowns)

        cycle = StateCycle(self._fluid, [i for i in range(self._graph.points)])
        for i in range(self._graph.points):
            for j in range(len_knowns):
                cycle[i, knowns[j]] = x0[i * len_knowns + j]
                cycle[i, Property.T.value] = 500.0
                cycle[i, Property.Y.value] = 1000.0

        cycle.calculate_properties(props=('P', 'T'))
        x0_complete = cycle._data

        sol_thermo = self._iterate_thermo(x0_complete, xtol=tol, verbose=verbose)

        bounds = self._graph.points * [(1.0, 1000.0)]
        sol_conserv = minimize(
            self._iterate,
            bounds=bounds,
            x0=x0_complete[:, Property.Y.value],
            args=(verbose,),
            method='L-BFGS-B',
            # options={'ftol': (2.0 * tol ** 2.0) / 10, 'maxiter': 1000}
            options={'ftol': 0.0, 'gtol': 0.0, 'maxiter': 100}
        )

        sol_thermo[:, Property.Y.value] = sol_conserv.x * 1000.0 / np.max(sol_conserv.x)
        self._graph.states._data = sol_thermo

        if verbose >= 3:
            print(sol_conserv)

        # Post-processing
        for part in self._graph.nodes.values():
            y_part = sum(self._graph.get_state((inlet.label, part.label))[Property.Y.value]
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

        return self._graph.states, sol_conserv.fun

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
