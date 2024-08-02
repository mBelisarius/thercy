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
            # inlets_state = {p.label: self._graph.get_state((p.label, part.label)) for p in part.inlet_parts}
            # outlets_state = {p.label: self._graph.get_state((part.label, p.label)) for p in part.outlet_parts}
            inlets = [p.label for p in part.inlet_parts]
            outlets = [p.label for p in part.outlet_parts]
            residual[2 * i:2 * i + 2] = self._graph[part.label].solve_conserv(self._graph, inlets, outlets)

        return residual

    def _iterate(self, x_states: np.ndarray, xtol=1e-4, verbose=0):
        cycle = StateCycle(self._fluid, [i for i in range(self._graph.points)])
        for i in range(self._graph.points):
            cycle[i, Property.P.value] = x_states[i * 3 + 0]
            cycle[i, Property.H.value] = x_states[i * 3 + 1]
            cycle[i, Property.Y.value] = x_states[i * 3 + 2]

        cycle.calculate_properties(props=('P', 'H'))
        x_complete = cycle._data
        y = x_complete[:, -1]

        sol_thermo = self._iterate_thermo(x_complete, xtol=xtol, verbose=verbose)
        # We already have sol_thermo == self._graph.states._data, so no need to update
        residual = self._equation_conserv(y)

        residual_mass = norm_l2(residual[0::2], rescale=True)
        residual_energy = norm_l2(residual[1::2], rescale=True)
        ressum = residual_energy + 1.0e6 * residual_mass

        if verbose >= 3:
            print(f"{'Rankine._iterate_conserv : ':40s}"
                  f"{ressum:.3e} | {residual_mass:.3e} | {residual_energy:3e}")

        return ressum

    def solve(self, x0, knowns=None, tol=1e-4, verbose=0):
        if knowns is None:
            raise ValueError('At least one property must be known for every state')

        len_props = len(Property)
        len_props_input = len(knowns)
        len_exclude = len_props - len_props_input

        t0 = 25.0 + 273.15
        p0 = 101.325e3
        x0_complete = np.array(len(x0) * [
            t0,
            p0,
            PropsSI('D', 'T', t0, 'P', p0, self._fluid),
            PropsSI('H', 'T', t0, 'P', p0, self._fluid),
            PropsSI('S', 'T', t0, 'P', p0, self._fluid),
            PropsSI('Q', 'T', t0, 'P', p0, self._fluid),
            1000.
        ])

        for i in range(0, len(x0), len_props_input):
            new_index = i + (i // len_props_input) * len_exclude
            indices = [new_index + PropertyInfo.get_intkey(k) for k in knowns]
            x0_complete[indices] = x0[i:i + len_props_input]

        bounds = self._graph.points * [(1.0e3, 100.0e6), (1.0e3, 100.0e6), (1.0, 1000.0)]
        indexes_use = np.array(
            [[i * len_props + Property.P.value, i * len_props + Property.H.value, i * len_props + Property.Y.value] for
             i in range(len(x0))], dtype=int).flatten()
        sol_conserv = minimize(
            self._iterate,
            bounds=bounds,
            x0=x0_complete[indexes_use],
            args=(tol, verbose,),
            method='L-BFGS-B',
            options={'ftol': tol / 10, 'maxiter': 1000}
        )

        sol_x = sol_conserv.x
        cycle = StateCycle(self._fluid, [i for i in range(self._graph.points)])
        for i in range(self._graph.points):
            cycle[i, Property.P.value] = sol_x[i * 3 + 0]
            cycle[i, Property.H.value] = sol_x[i * 3 + 1]
            cycle[i, Property.Y.value] = sol_x[i * 3 + 2]

        cycle.calculate_properties(props=('P', 'H'))

        for index in range(self._graph.points):
            self._graph.states[index] = cycle._data[index]

        print(sol_conserv)
        print(self._graph.states)

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
