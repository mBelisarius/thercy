import numpy as np
from scipy.optimize import minimize, root

from thercy.constants import PartType, Property
from thercy.state import StateGraph
from thercy.utils import norm_l1, norm_l2, norm_lmax, norm_lp


class Rankine:
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

    def _equation_conserv(self, y: np.ndarray):
        residual = np.zeros(2 * len(self._graph))

        for index in range(self._graph.points):
            self._graph.states[index]['Y'] = y[index]

        for i, part in enumerate(self._graph.nodes.values()):
            inlets_state = {p.label: self._graph.get_state((p.label, part.label)) for p in part.inlet_parts}
            outlets_state = {p.label: self._graph.get_state((part.label, p.label)) for p in part.outlet_parts}
            residual[2 * i:2 * i + 2] = self._graph[part.label].solve_conserv(inlets_state, outlets_state)

        return residual

    def _iterate_conserv_scalar(self, x: np.ndarray, verbose=0):
        residual = self._equation_conserv(x)
        residual_mass = norm_l2(residual[0::2])
        residual_energy = norm_l2(residual[1::2])

        if verbose >= 3:
            print(f"{'Rankine._iterate_conserv : ':40s}"
                  f"{residual_energy:3e} | {np.sqrt(residual_energy / len(self._graph)):3e} | "
                  f"{residual_mass:.3e} | {np.sqrt(residual_mass / len(self._graph)):.3e}")

        return residual_energy + 1e9 * residual_mass

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
                  f"fun = {sol.fun}, nfev={sol.nfev}")

        return sol.x

    def _iterate(self, x0_fraction: np.ndarray, x0_states: np.ndarray, update_states=False, xtol=1e-4, verbose=0):
        x0_states_ = x0_states.copy()
        x0_states_[len(Property) - 1::len(Property)] = x0_fraction
        sol_thermo = self._iterate_thermo(x0_states_, xtol=xtol, verbose=verbose)

        bounds = self._graph.points * [(1.0, 1000.0)]
        sol_conserv = minimize(
            self._iterate_conserv_scalar,
            bounds=bounds,
            x0=x0_fraction,
            args=(verbose,),
            method='L-BFGS-B',
            options={'ftol': xtol / 10, 'maxiter': 1000}
        )

        if update_states:
            x0_states = x0_states_

        if verbose >= 2:
            # residual = self._equation_conserv(sol_conserv.x)
            # residual_mass = norm_l2(residual[0::2])
            # residual_energy = norm_l2(residual[1::2])
            # print(f"{'Rankine._iterate : ':40s}"
            #       f"{residual_energy:3e} | {np.sqrt(residual_energy / 11):3e} | "
            #       f"{residual_mass:.3e} | {np.sqrt(residual_mass / 11):.3e}")
            pass

        return x0_fraction - sol_conserv.x

    def solve(self, x0, knowns=None, xtol=1e-4, verbose=0):
        if knowns is None:
            raise ValueError('At least one property must be known for every state')

        len_props = len(Property)
        len_props_input = len(knowns)
        len_exclude = len_props - len_props_input
        x0_complete = np.ones(len_props * len(x0) // len_props_input)

        for i in range(0, len(x0), len_props_input):
            new_index = i + (i // len_props_input) * len_exclude
            x0_complete[new_index:new_index + len_props_input] = x0[i:i + len_props_input]

        if 'Y' in knowns:
            x0_fraction = x0_complete[len_props - 1::len_props]
            x0_fraction *= 1000.0 / np.max(x0_fraction)
        else:
            x0_fraction = np.full(self._graph.points, 1000.0)
            x0_complete[len_props - 1::len_props] = x0_fraction

        # Solve for the conservation laws
        sol_conserv = root(
            self._iterate,
            x0_fraction,
            method='df-sane',
            args=(x0_complete, True, xtol, verbose),
            options={'sigma_0': 1e-4, 'fatol': xtol, 'ftol': 0.0, 'maxfev': 100}
        )

        # Process the solution, guarantees that graph.states contains the
        # correct values
        sol_x = x0_complete.copy()
        sol_x[len_props - 1::len_props] = sol_conserv.x * 1000.0 / np.max(sol_conserv.x)
        sol_thermo = root(
            self._equation_thermo,
            sol_x,
            method='df-sane',
            options={'fatol': xtol, 'maxfev': 10}
        )
        sol_x = sol_thermo.x
        for index in range(self._graph.points):
            index_begin = index * len_props
            index_end = index_begin + len_props
            self._graph.states[index].from_array(sol_x[index_begin:index_end])

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
