import numpy as np
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots import StateContainer
from itertools import combinations

from thercy.constants import Property, PropertyInfo
from thercy.utils import list_like


class StateCycle:
    """
         Class StateCycle:
    """
    def __init__(self, fluid, data=None):
        """
        Initialize a `StateCycle` instance.

        Parameters
        ----------
        fluid : str
            Fluid composition or mixture name.
        data : dict[str: numpy.ndarray] | list[str], optional
            Dictionary containing state points of the cycle. Default: {}
        """
        if isinstance(data, (list, tuple, np.ndarray)):
            data = {key: self.new_empty_state(1) for key in data}

        if isinstance(data, dict):
            data_inspect = next(iter((data.values())))
            if isinstance(data_inspect, (list, np.ndarray)):
                if not np.all([d.size == len(Property) for d in data.values()]):
                    raise ValueError('All states must contain all properties')
            elif isinstance(data_inspect, dict):
                aux = {k: self.new_empty_state() for k, v in data.values()}
                for k in data.keys():
                    for prop in Property:
                        prop_key = PropertyInfo.get_strkey(prop.name)
                        if prop_key in data[k].keys():
                            aux[k][prop.value] = data[k][prop_key]
                data = aux
            else:
                raise TypeError(f"Non supported state type {type(data_inspect)}")

        if data is None:
            data = {}

        self._matrix = np.array([data[k] for k in data.keys()], dtype=np.float64)
        self._data_keys = {k: i for i, k in enumerate(data.keys())}
        self._fluid: str = fluid

    def __len__(self):
        """Get the number of state points in the cycle."""
        return len(self._data_keys)

    def __iter__(self):
        """Iterator over state points in the cycle."""
        # TODO: Test
        return iter(self._data_keys)

    def __str__(self):
        """String representation of the cycle's state points."""
        out = ''

        # Retrieve the keys of the first StatePoint to get property labels
        row = [f"{'State':>5s}"]
        for prop in Property:
            label = f"{PropertyInfo.label(prop)} ({PropertyInfo.unit(prop)})"
            row.append(f"{label:>16s}")
        out += '  '.join(row) + '\n'

        for k, i in self._data_keys.items():
            row = [f"{str(k):>5s}"]
            for prop in Property:
                value = self._matrix[i][prop.value]
                if value is not None:
                    row.append(f"{value:16.3f}")
                else:
                    row.append(f"{'-':>16s}")
            out += '  '.join(row) + '\n'

        return out

    def __getitem__(self, key):
        """
        Get a state point by keys.

        Parameters
        ----------
        key : str | tuple
            Key to access the state point.

        Returns
        -------
        StatePoint | float
            State point corresponding to the key.

        Raises
        ------
        IndexError if the index has an invalid length.
        """
        if list_like(key):
            len_var = len(key)
            if len_var == 1:
                return self._get_point(key[0])
            elif len_var == 2:
                return self._get_float(key)
            else:
                raise IndexError("Too many indexes to unpack.")

        return self._get_point(key)

    def _get_float(self, key):
        """
        Get a state point property value by key pair.

        Parameters
        ----------
        key : tuple
            Key pair to access the state point.

        Returns
        ------
        value : float
        """
        index_state = self._data_keys[key[0]]
        index_prop = PropertyInfo.get_intkey(key[1])
        return self._matrix[index_state, index_prop]

    def _get_point(self, key):
        """
        Get a state point value by key.

        Parameters
        ----------
        key : str
            Key to access the state point.

        Returns
        ------
        value : StatePoint
        """
        return self._matrix[self._data_keys[key]]

    def __setitem__(self, key, value):
        """
        Set a state point by key or key pair.

        Parameters
        ----------
        key : str | tuple
            Key to access the state point.
        value : numpy.ndarray | float
            State point or float value to assign.

        Raises
        ------
        IndexError if the index has an invalid length.
        """
        if list_like(key):
            len_var = len(key)
            if len_var == 1:
                self._set_point(key[0], value)
            elif len_var == 2:
                self._set_float(key, value)
            else:
                raise IndexError("Index with invalid length.")
        else:
            self._set_point(key, value)

    def _set_float(self, key, value):
        """
        Set a state point value by key pair.

        Parameters
        ----------
        key : tuple
            Key pair to access the state point.
        value : float
            Float value to assign.

        Raises
        ------
        TypeError if the value type is not numeric.
        """
        if not isinstance(value, (int, float)):
            raise TypeError('Value is not a numeric type')

        if key[0] not in self._data_keys.keys():
            self._data_keys[key[0]] = len(self._data_keys)
            if len(self._matrix) > 0:
                self._matrix = np.append(self._matrix, self.new_empty_state(2), axis=0)
            else:
                self._matrix = self.new_empty_state(2)

        index_state = self._data_keys[key[0]]
        index_prop = PropertyInfo.get_intkey(key[1])
        self._matrix[index_state, index_prop] = value

    def _set_point(self, key, value):
        """
        Set a state point by key.

        Parameters
        ----------
        key : str
            Key to access the state point.
        value : numpy.ndarray
            State point to assign.

        Raises
        ------
        TypeError if the value type is invalid.
        """
        if value is None:
            value = self.new_empty_state(2)

        if value.size != len(Property):
            raise ValueError('State must contain all properties')

        if key not in self._data_keys.keys():
            self._data_keys[key] = len(self._data_keys)
            if len(self._matrix) > 0:
                self._matrix = np.append(self._matrix, self.new_empty_state(2), axis=0)
            else:
                self._matrix = self.new_empty_state(2)

        self._matrix[self._data_keys[key]] = value

    @property
    def fluid(self):
        """Get the fluid composition or mixture name."""
        return self._fluid

    @property
    def matrix(self):
        """Get the matrix of thermodynamic states."""
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        """
        Set the matrix of states.

        Parameters
        ----------
        value : numpy.ndarray
            Matrix of states.
        """
        if isinstance(value, np.ndarray):
            if value.ndim != 2:
                raise ValueError('Matrix must have 2 dimensions')
            if value.shape[1] != len(Property):
                raise ValueError('Matrix columns must match number of properties')
            self._matrix = value
        else:
            raise TypeError('Matrix must be a numpy array')

    @property
    def first(self):
        """Get the first state in the cycle."""
        if self._matrix:
            return self._matrix[0]

    @property
    def last(self):
        """Get the last state in the cycle."""
        if self._matrix:
            return self._matrix[-1]

    @staticmethod
    def new_empty_state(dim=1):
        """
        Generate an empty array represeting a state.

        Parameters
        ----------
        dim : int, optional
            The dimension of the state. Default: 1

        Returns
        -------
        numpy.ndarray
            An empty state array filled with NaN values.

        Raises
        ------
        ValueError
            If the dimension is not 1 or 2.
        """
        if dim == 1:
            return np.full(len(Property), np.nan, dtype=np.float64)
        elif dim == 2:
            return np.full((1, len(Property)), np.nan, dtype=np.float64)
        else:
            raise ValueError('Invalid dimension')

    def to_state_container(self):
        """
        Convert the cycle's data to CoolProp.Plots.StateContainer.

        Returns
        -------
        CoolProp.Plots.StateContainer
            State container.
        """
        container = StateContainer()

        for i, k in enumerate(self._data_keys.keys()):
            for prop in Property:
                if prop.name not in ['Y']:
                    container[i, prop.name] = self[k, prop.name]

        return container

    @staticmethod
    def calculate_props(state, fluid, prop1=None, prop2=None, calc=None, exclude=None):
        """
        Calculate missing properties based on known ones.

        Parameters
        ----------
        state : numpy.ndarray
            State to be calculated.
        fluid : str
            Fluid composition or mixture name.
        prop1 : Property | str | int, optional
            First known property. Default: None
        prop2 : Property | str | int, optional
            Second known property. Default: None
        calc : list[str], optional
            Properties to calculate. Default: all
        exclude : list[str], optional
            Properties to exclude from calculation. Default: None
        """
        # TODO: Move definition of not thermodynamic properties to `constants`
        not_thermo = ['Y']

        if calc is None:
            calc = [prop.name for prop in Property
                    if prop.name not in (prop1, prop2)]

        if exclude is None:
            exclude = []

        knowns = {Property(i).name: v for i, v in enumerate(state) if v is not None and not np.isnan(v)}

        if prop1 is not None and prop2 is not None:
            pairs = [(prop1, prop2)]
        elif prop1 is not None:
            pairs = [(prop1, p) for p in knowns.keys() if p != prop1 and p not in not_thermo]
        else:
            pairs = combinations(knowns.keys(), 2)

        to_calc = [prop for prop in calc
                   if prop not in not_thermo
                   and prop not in exclude]

        while len(to_calc) > 0:
            prop = to_calc.pop()

            for pair in pairs:
                if prop in pair:
                    continue

                try:
                    index_prop = PropertyInfo.get_intkey(prop)
                    state[index_prop] = PropsSI(
                        prop,
                        pair[0],
                        knowns[pair[0]],
                        pair[1],
                        knowns[pair[1]],
                        fluid
                    )

                except ValueError as e:
                    pass

    def calculate_properties(self, key=None, props=None):
        """
        Calculate properties for all state points in the cycle.

        Parameters
        ----------
        key : str | list[str], optional
            Key of the states to be calculated. Default: all
        props : tuple[Property | str | int], optional
            Tuple of known properties to use in the computation. Default: None
        """
        if key is None:
            key = self._data_keys.keys()

        if props is None:
            props = (None, None)

        if not list_like(key):
            key = [key]

        for k in key:
            index_state = self._data_keys[k]
            self.calculate_props(self._matrix[index_state], self._fluid, *props)

    def constant_properties(self, key1, key2, tol=1e-7):
        """
        Get properties with constant values between two state points.

        Parameters
        ----------
        key1 : str
            Key of the first state point.
        key2 : str
            Key of the second state point.
        tol : float, optional
            Tolerance for constant property comparison. Default: 1e-7

        Returns
        -------
        list[Property]
            List of constant properties.
        """
        index_state1 = self._data_keys[key1]
        index_state2 = self._data_keys[key2]
        return [prop for prop in Property if abs(self[index_state1, prop.value] - self[index_state2, prop.value]) <= tol]

    def cloud_points(self, n=50, close_envelope=False, precise=False):
        # TODO: Fix
        """
        Generate cloud points between consecutive state points.

        Parameters
        ----------
        n : int, optional
            Number of points between each pair of state points. Default: 50
        close_envelope : bool, optional
            Close the cycle envelope by connecting the last state to the first
            state. Default: False
        precise : bool, optional
            Use a more precise version of interpolation by enthalpy instead of
            temperature, slower. Default: False

        Returns
        -------
        StateCycle
            State cycle containing the cloud points.
        """
        cloud = StateCycle(self._fluid)

        for i, (key1, state1) in enumerate(self._matrix.items()):
            # Close the cycle envelope
            if i + 1 < len(self._matrix):
                key2, state2 = list(self._matrix.items())[i + 1]
            elif close_envelope:
                key2, state2 = list(self._matrix.items())[0]
            else:
                break

            prop_x = next((prop for prop in self.constant_properties(key1, key2)
                           if prop.name not in ('S', 'Q', 'Y')), 'H' if precise else 'T')

            s_diff = state2['S'] - state1['S']
            y_diff = state2[prop_x] - state1[prop_x]

            s = np.linspace(state1['S'], state2['S'], n - 1)
            s = np.append(s, state2['S'])

            x = (y_diff / s_diff) * (s - state1['S']) + state1[prop_x] if abs(s_diff) > 1e-7 \
                else np.linspace(state1[prop_x], state2[prop_x], n)

            for j in range(n):
                cloud_key = j + i * n
                cloud[cloud_key, 'S'] = s[j]
                cloud[cloud_key, prop_x] = x[j]
                cloud[cloud_key].properties()

        return cloud
