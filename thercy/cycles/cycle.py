import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize, root

from thercy.constants import PartType, Property, PropertyInfo
from thercy.state import StateCycle, StateGraph
from thercy.utils import norm_l1, norm_l2, norm_lmax, norm_lp


class CycleResult:
    def __init__(self, x, success, fun, nit):
        self._x = x
        self._success = success
        self._fun = fun
        self._nit = nit

    def __str__(self):
        return (f"{'x':>20}: {self._x}\n"
                f"{'success':>20}: {self._success}\n"
                f"{'fun':>20}: {self._fun}\n"
                f"{'nit':>20}: {self._nit}")

    @property
    def x(self):
        return self._x

    @property
    def success(self):
        return self._success

    @property
    def fun(self):
        return self._fun

    @property
    def nit(self):
        return self._nit


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
    def bwr(self):
        return self._work_pumps / -self._work_turbines

    @property
    def cycle(self):
        return self._graph.states

    @property
    def efficiency(self):
        return self.work / self.heat_input

    @property
    def graph(self):
        return self._graph

    @property
    def heat_input(self):
        return self._heat_input

    @property
    def heat_output(self):
        return -self._heat_output

    def massflow(self, power):
        return power / self.work

    @property
    def states(self):
        return self._graph.states

    @property
    def work(self):
        return -(self._work_pumps + self._work_turbines)

    def _equation_thermo(self, x: np.ndarray):
        len_props = len(Property)
        len_states = self._graph.points
        residual = np.zeros(x.size)

        for index in range(len_states):
            self._graph.states[index] = x[index]

        for part in self._graph.nodes.values():
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

    def _iterate_thermo(self, x0: np.ndarray, fatol=1e-4, maxfev=10, verbose=0):
        sol = root(
            self._equation_thermo,
            x0,
            method='df-sane',
            options={'fatol': fatol, 'maxfev': maxfev}
        )

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
        # We already have sol_thermo == self._graph.states.matrix, so no need to update
        y *= 1000.0 / np.max(y)
        residual = self._equation_conserv(y)

        residual_mass = norm_l2(residual[0::2], rescale=True)
        residual_energy = norm_l2(residual[1::2], rescale=True)
        ressum = residual_energy ** 2.0 + (1.0e6 * residual_mass) ** 2.0

        if verbose >= 3:
            print(f"{'Rankine._iterate_conserv : ':40s}"
                  f"{ressum:.3e} | {residual_mass:.3e} | {residual_energy:3e}")

        return ressum

    def solve(self, x0, x0props, fatol=1e-4, verbose=0):
        len_knowns = len(x0props)

        if len_knowns < 2:
            raise ValueError("x0 must have at least two property guesses.")

        cycle = StateCycle(self._fluid, [i for i in range(self._graph.points)])
        for i in range(self._graph.points):
            for j in range(len_knowns):
                cycle[i, Property.Y.value] = 1000.0
                if x0.ndim == 1:
                    cycle[i, x0props[j]] = x0[i * len_knowns + j]
                elif x0.ndim == len_knowns:
                    cycle[i, x0props[j]] = x0[i, j]
                elif x0.ndim == self._graph.points:
                    cycle[i, x0props[j]] = x0[j, i]
                else:
                    raise ValueError("x0 must be in a valid shape.")

        cycle.calculate_properties(props=x0props)

        sol_thermo = self._iterate_thermo(cycle.matrix, fatol=fatol, verbose=verbose)
        cycle.matrix = sol_thermo

        bounds = self._graph.points * [(1.0, 1000.0)]
        sol_conserv = minimize(
            self._iterate,
            bounds=bounds,
            x0=cycle.matrix[:, Property.Y.value],
            args=(verbose,),
            method='L-BFGS-B',
            # options={'ftol': (2.0 * tol ** 2.0) / 10, 'maxiter': 1000}
            options={'ftol': 0.0, 'gtol': 0.0, 'maxiter': 100}
        )

        sol_thermo[:, Property.Y.value] = sol_conserv.x * 1000.0 / np.max(sol_conserv.x)
        self._graph.states.matrix = sol_thermo
        cycle.matrix = sol_thermo
        residual = np.sqrt(sol_conserv.fun / 2.0)
        sol = CycleResult(cycle, residual < fatol, residual, sol_conserv.nit)

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

        return sol
