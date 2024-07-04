import numpy as np
from CoolProp.Plots import StateContainer

from thercy.constants import Property, PropertyInfo
from thercy.utils import list_like

from .state_point import StatePoint


class StateCycle:
    """
    Thermodynamic state cycle composed of multiple state points.

    Properties
    ----------
    fluid : str
        Fluid composition or mixture name.
    data : dict
        Dictionary containing keyed state points of the cycle.

    """
    def __init__(self, fluid, data=None):
        """
        Initialize a StateCycle instance.

        Parameters
        ----------
        fluid : str
            Fluid composition or mixture name.
        data : dict[any: StatePoint], optional
            Dictionary containing state points of the cycle. Default: {}

        """
        if data is None:
            data = {}

        self._data: dict[any: StatePoint] = data
        self._fluid: str = fluid

    def __len__(self):
        """Get the number of state points in the cycle."""
        return len(self._data)

    def __iter__(self):
        """Iterator over state points in the cycle."""
        return iter(self._data)

    def __str__(self):
        """String representation of the cycle's state points."""
        out = ''

        # Retrieve the keys of the first StatePoint to get property labels
        row = [f"{'State':>5s}"]
        for prop in Property:
            label = f"{PropertyInfo.label(prop)} ({PropertyInfo.unit(prop)})"
            row.append(f"{label:>16s}")
        out += '  '.join(row) + '\n'

        for k in self._data.keys():
            row = [f"{str(k):>5s}"]
            for prop in Property:
                value = self._data[k][prop.name]
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
        key : any | tuple
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
                raise IndexError("Index with invalid length.")

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
        return self._data[key[0]][key[1]]

    def _get_point(self, key):
        """
        Get a state point value by key.

        Parameters
        ----------
        key : any
            Key to access the state point.

        Returns
        ------
        value : StatePoint

        """
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Set a state point by key or key pair.

        Parameters
        ----------
        key : any | tuple
            Key to access the state point.
        value : StatePoint | float
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

        if key[0] not in self._data:
            self._data[key[0]] = StatePoint(self._fluid)

        self._data[key[0]][key[1]] = value

    def _set_point(self, key, value):
        """
        Set a state point by key.

        Parameters
        ----------
        key : any
            Key to access the state point.
        value : StatePoint
            State point to assign.

        Raises
        ------
        TypeError if the value type is invalid.

        """
        if not isinstance(value, StatePoint):
            raise TypeError('Value is not a StatePoint')

        self._data[key] = value

    @property
    def fluid(self):
        """Get the fluid composition or mixture name."""
        return self._fluid

    @property
    def first(self):
        """Get the first state point in the cycle."""
        if self._data:
            return next(iter(self._data.values()))

    @property
    def last(self):
        """Get the last state point in the cycle."""
        if self._data:
            return next(iter(reversed(self._data.values())))

    def to_state_container(self):
        """
        Convert the cycle's data to CoolProp.Plots.StateContainer.

        Returns
        -------
        CoolProp.Plots.StateContainer
            State container.

        """
        container = StateContainer()

        for i, k in enumerate(self._data.keys()):
            for prop in Property:
                if prop.name not in ['Y']:
                    container[i, prop.name] = self[k, prop.name]

        return container

    def calculate_properties(self):
        """Calculate properties for all state points in the cycle."""
        for sp in self._data.values():
            sp.properties()

    def constant_properties(self, key1, key2, tol=1e-7):
        """
        Get properties with constant values between two state points.

        Parameters
        ----------
        key1 : any
            Key of the first state point.
        key2 : any
            Key of the second state point.
        tol : float, optional
            Tolerance for constant property comparison. Default: 1e-4

        Returns
        -------
        list[Property]
            List of constant properties.

        """
        return [prop for prop in Property if abs(self[key1, prop.name] - self[key2, prop.name]) <= tol]

    def cloud_points(self, n=50, close_envelope=False, precise=False):
        """
        Generate cloud points between consecutive state points.

        Parameters
        ----------
        n : int, optional
            Number of points between each pair of state points.
            Default: 50
        close_envelope : bool, optional
            Close the cycle envelope by connecting the last state to the
            first state. Default: False
        precise : bool, optional
            Use a more precise version of interpolation by enthalpy
            instead of temperature, slower. Default: False

        Returns
        -------
        StateCycle
            State cycle containing the cloud points.

        """
        cloud = StateCycle(self._fluid)

        for i, (key1, state1) in enumerate(self._data.items()):
            # Close the cycle envelope
            if i + 1 < len(self._data):
                key2, state2 = list(self._data.items())[i + 1]
            elif close_envelope:
                key2, state2 = list(self._data.items())[0]
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
