import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import lsmr, lsqr, bicgstab

from thercy.constants import PartType, Property, PropertyInfo
from thercy.state import StateCycle, StateGraph
from thercy.utils import norm_l1, norm_l2, norm_lmax, norm_lp


class CycleResult:
    """
    Represents the cycle result.

    Properties
    ----------
    x : numpy.ndarray
        The solution of the cycle.
    success : bool
        Whether or not the solver exited successfully.
    res : float
        Residual of the result.
    nit : int
        Number of iterations performed by the solver.

    Notes
    -----
    `CycleResult` may not have all attributes listed here.
    """
    def __init__(self, x, success, fun, nit):
        self._x = x
        self._success = success
        self._res = fun
        self._nit = nit

    def __str__(self):
        return (f"{'x':>20}: {self._x}\n"
                f"{'success':>20}: {self._success}\n"
                f"{'fun':>20}: {self._res}\n"
                f"{'nit':>20}: {self._nit}")

    @property
    def x(self):
        """Get the solution of the cycle."""
        return self._x

    @property
    def success(self):
        """Get whether or not the solver exited successfully."""
        return self._success

    @property
    def res(self):
        """Get the residual of the result."""
        return self._res

    @property
    def nit(self):
        """Get the number of iterations performed by the solver."""
        return self._nit


class Cycle:
    """
    Thermodynamic cycle solver.
    """
    def __init__(self, fluid, parts, fraction_base=1000.0):
        """
        Initialize a `Cycle` instance.

        Parameters
        ----------
        fluid : str
            Fluid to be used in the cycle.
        parts : StateGraph
            `StateGraph` containing the parts and connections of the cycle.
        fraction_base : float, optional
            The maximum value for fractions to be used for the cycle.
            If 1.0, then fractions will be a percentage. Default: 1000.0
        """
        self._fluid: str = fluid
        self._graph: StateGraph = parts
        self._fraction_base = fraction_base
        self._heat_input: float = 0.0
        self._heat_output: float = 0.0
        self._work_pumps: float = 0.0
        self._work_turbines: float = 0.0

    def __len__(self):
        """Get the number of parts in the cycle."""
        return len(self._graph)

    def __str__(self):
        """String representation of the state cycle."""
        return str(self._graph)

    @property
    def bwr(self):
        """Get the BWR of the cycle."""
        return self._work_pumps / -self._work_turbines

    @property
    def efficiency(self):
        """Get the efficiency of the cycle."""
        return self.work / self.heat_input

    @property
    def graph(self):
        """Get the `StateGraph` associated with this."""
        return self._graph

    @property
    def heat_input(self):
        """Get the heat inputed into the cycle."""
        return self._heat_input

    @property
    def heat_output(self):
        """Get the heat outputed from the cycle."""
        return -self._heat_output

    def massflow(self, power):
        """
        Get the mass flow rate necessary to achieve the given `power` output.

        Parameters
        ----------
        power : float
            Power required by the cycle.

        Returns
        -------
        float
            Mass flow rate necessary to achieve `power`.
        """
        return power / self.work

    @property
    def states(self):
        """Get the `StateCycle` associated with this."""
        return self._graph.states

    @property
    def work(self):
        """Get the work done by the cycle."""
        return -(self._work_pumps + self._work_turbines)

    def _equation_thermo(self, x):
        """
        Calculates the residual of an equation system for thermochemical equilibrium.

        Parameters
        ----------
        x : numpy.ndarray
            An array containing the unknown values of the variables in the equation system.

        Returns
        -------
        residual : numpy.ndarray
            An array containing the residuals of the equation system.

        Note
        ----
        The equation system represents the thermochemical equilibrium of a
        process. It solves for the unknown values of the variables based on the
        current state of the graph. The equation system is solved using an
        iterative method.

        The residuals of the equation system are calculated by comparing the
        solved values with the input values in the ``x`` array.
        """
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
                    if value[prop.value] is not None and not np.isnan(value[prop.value]):
                        self._graph.states[edge_index, prop.value] = value[prop.value]

        for index in range(len_states):
            index_begin = index * len_props
            index_end = index_begin + len_props
            residual[index_begin:index_end - 1] = (x[index] - self._graph.states[index])[:-1]

        return residual

    def _iterate_thermo(self, x0, fatol=1e-4, maxfev=10, verbose=0):
        """
        Iterates the thermodynamic equations to find the solution for a given
        initial guess ``x0``.

        Parameters
        ----------
        x0 : numpy.ndarray
            The initial guess for the solution.
        fatol : float, optional
            The absolute error tolerance for the solution. Default: 1e-4
        maxfev : int, optional
            The maximum number of iterations allowed. Default: 10
        verbose : int, optional
            The level of verbosity. Set to 0 for no output, higher values for
            more output. Default: 0

        Returns
        -------
        numpy.ndarray
            The solution vector.
        """
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
                  f"L2(fun) = {norm_l2(sol.fun, normalize=True)}, nfev={sol.nfev}")

        return sol.x

    def _equation_conserv(self, y):
        """
        This method performs an equation conservation calculation and returns
        the resulting residual.

        Parameters
        ----------
        y : numpy.ndarray
            The array of values representing the state fractions.

        Returns
        -------
        numpy.ndarray
            The array of residuals after the equation conservation calculation.

        """
        residual = np.zeros(2 * len(self._graph))

        for index in range(self._graph.points):
            self._graph.states[index, 'Y'] = y[index]

        for i, part in enumerate(self._graph.nodes.values()):
            inlets = [p.label for p in part.inlet_parts]
            outlets = [p.label for p in part.outlet_parts]
            residual[2 * i:2 * i + 2] = self._graph[part.label].solve_conserv(self._graph, inlets, outlets)

        return residual

    def _iterate_conserv(self, y0=None, atol=1e-4):
        """
        Solve a system of linear equations representing the conservation laws.

        Parameters
        ----------
        y0 : numpy.ndarray, optional
            The initial guess. Default: None
        atol : float, optional
            The absolute tolerance for convergence. The iteration will stop when
            the residual norm is smaller than this value. Default: 1e-4

        Returns
        -------
        x : ndarray
            The solution vector to the system of linear equations.
        normr : float
            The norm of the residual.
        itn : int
            The number of iterations performed to reach the solution.
        """
        if y0 is None:
            y0 = np.full(self._graph.points, self._fraction_base)

        parts_map = {i: label for i, label in enumerate(self._graph.nodes)}
        coeffs_mass = np.zeros((self._graph.points, self._graph.points), dtype=np.float64)
        coeffs_energy = np.zeros((self._graph.points, self._graph.points), dtype=np.float64)

        index_mass = 0
        index_energy = 0
        for label in parts_map.values():
            part = self._graph[label]
            conns = part.connections
            for con in conns:
                for inlet in con.inlets:
                    edge_index = self._graph.get_edge_index((inlet.label, label))
                    coeffs_mass[index_mass, edge_index] = -1.0e6
                for outlet in con.outlets:
                    edge_index = self._graph.get_edge_index((label, outlet.label))
                    coeffs_mass[index_mass, edge_index] = 1.0e6
                index_mass += 1

            for inlet in part.inlet_parts:
                edge_index = self._graph.get_edge_index((inlet.label, label))
                coeffs_energy[index_energy, edge_index] = -self._graph.states.matrix[edge_index, Property.H.value]
            for outlet in part.outlet_parts:
                edge_index = self._graph.get_edge_index((label, outlet.label))
                coeffs_energy[index_energy, edge_index] = self._graph.states.matrix[edge_index, Property.H.value]
            index_energy += 1

        coeffs = coeffs_mass

        non_redundant = np.argsort([np.count_nonzero(coeffs_energy[i, :]) for i in range(len(self._graph))], kind='stable')
        necessary_count = self._graph.points - np.count_nonzero([np.count_nonzero(coeffs_mass[i, :]) for i in range(self._graph.points)])
        if necessary_count > 0:
            necessary_indexes = non_redundant[-necessary_count:]
            coeffs[-necessary_count:] = coeffs_energy[necessary_indexes]

        rhs = np.zeros(self._graph.points)
        for i in range(necessary_count):
            label = parts_map[necessary_indexes[i]]
            rhs[len(self._graph) + i] = self._graph[label].deltaH

        # Define the boundary condition by the penalty method
        # y0 = self._fraction_base
        coeffs[0, 0] = 1.0e16
        rhs[0] = self._fraction_base * 1.0e16

        # TODO: Determine the best solver
        # x, res, rnk, s = lstsq(coeffs, rhs, check_finite=False)
        x, itn = bicgstab(coeffs, rhs, x0=y0, rtol=0.0, atol=atol)
        np.clip(x, a_min=1.0e-7, a_max=None, out=x)  # No negative mass flow allowed
        normr = np.linalg.norm(coeffs @ x - rhs)

        return x, normr, itn

    def solve(self, x0, x0props, maxiter=10, fatol=1e-4, verbose=0):
        """
        Solves the thermodynamic cycle using the given initial conditions.

        Parameters
        ----------
        x0 : numpy.ndarray
            The initial condition for the property guesses. It can be a
            1-dimensional array if x0props has a length of 1, a 2-dimensional
            array if x0props has the same length as the number of rows in x0, or
            a 2-dimensional array if x0props has the same length as the number
            of columns in x0.
        x0props : list | tuple
            The list of property names corresponding to the columns of x0.
        maxiter : int, optional
            The maximum number of iterations to perform. Default: 10
        fatol : float, optional
            The convergence absolute tolerance for the thermodynamic solver.
            Default: 1e-4
        verbose : int, optional
            The level of verbosity. Set to 0 for no output, higher values for
            more output. Default: 0

        Returns
        -------
        CycleResult
            A `CycleResult` containing the results of the thermodynamic cycle
            solution.

        Raises
        ------
        ValueError
            If the length of x0props is less than 2.
            If the shape of x0 is not valid.
        """
        len_knowns = len(x0props)
        if len_knowns < 2:
            raise ValueError("x0 must have at least two property guesses.")

        cycle = StateCycle(self._fluid, [i for i in range(self._graph.points)])
        for i in range(self._graph.points):
            for j in range(len_knowns):
                cycle[i, Property.Y.value] = self._fraction_base
                if x0.ndim == 1:
                    cycle[i, x0props[j]] = x0[i * len_knowns + j]
                elif x0.ndim == len_knowns:
                    cycle[i, x0props[j]] = x0[i, j]
                elif x0.ndim == self._graph.points:
                    cycle[i, x0props[j]] = x0[j, i]
                else:
                    raise ValueError("x0 must be in a valid shape.")

        cycle.calculate_properties(props=x0props)

        for it in range(1, maxiter + 1):
            sol_thermo = self._iterate_thermo(cycle.matrix, fatol=fatol, verbose=verbose)
            self._graph.states.matrix = sol_thermo

            sol_conserv, res_conserv, nit_conserv = self._iterate_conserv(cycle.matrix[:, Property.Y.value])
            sol_thermo[:, Property.Y.value] = sol_conserv * self._fraction_base / np.max(sol_conserv)

            self._graph.states.matrix = sol_thermo
            cycle.matrix = sol_thermo

            if res_conserv < fatol:
                break

        sol_thermo = self._iterate_thermo(cycle.matrix, fatol=fatol, verbose=verbose)
        sol_thermo[:, Property.Y.value] = sol_conserv * self._fraction_base / np.max(sol_conserv)
        self._graph.states.matrix = sol_thermo

        residual = res_conserv
        sol = CycleResult(self._graph.states, residual < fatol, residual, it)

        # Post-processing
        self._heat_input = 0.0
        self._heat_output = 0.0
        self._work_pumps = 0.0
        self._work_turbines = 0.0

        for part in self._graph.nodes.values():
            match part.type:
                case PartType.CONDENSATOR:
                    self._heat_output += part.deltaH
                case PartType.HEAT_SOURCE:
                    self._heat_input += part.deltaH
                case PartType.PUMP:
                    self._work_pumps += part.deltaH
                case PartType.TURBINE:
                    self._work_turbines += part.deltaH

        return sol
