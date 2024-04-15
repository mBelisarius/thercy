import numpy as np
from CoolProp.CoolProp import PropsSI
from itertools import combinations

from thercy.constants import Property, PropertyInfo


class StatePoint:
    """
    Thermodynamic state point.

    Properties
    ----------
    fluid : str
        Fluid composition or mixture name.
    data : list
        State data corresponding to thermodynamic properties.

    """

    def __init__(self, fluid, data=None):
        """
        Initialize a StatePoint instance.

        Parameters
        ----------
        fluid : str
            Fluid composition or mixture name.
        data : dict, optional
            State data corresponding to thermodynamic properties.
            Default: {}

        """
        self._fluid: str = fluid

        if data is None:
            data = {}
        self._data: list = [data.get(p.name) for p in Property]

    def __iter__(self):
        """Iterator over state data."""
        return iter(self._data)

    def __str__(self):
        """String representation of the state data."""
        return str(self._data)

    def __getitem__(self, prop):
        """
        Get the value of a property.

        Parameters
        ----------
        prop : Property | str | int
            Property to retrieve.

        Returns
        -------
        value : int
            State value of the property.

        """
        return self._data[PropertyInfo.get_intkey(prop)]

    def __setitem__(self, prop, value):
        """
        Set the value of a property.

        Parameters
        ----------
        prop : Property | str | int
            Property to set.
        value : float
            Value to assign to the property.

        """
        self._data[PropertyInfo.get_intkey(prop)] = value

    @property
    def fluid(self):
        """Get the fluid composition or mixture name."""
        return self._fluid

    @property
    def data(self):
        """Get the state data corresponding to thermodynamic properties."""
        return self._data

    def clone(self):
        """Create a copy of the state point."""
        data = {p.name: self._data[p.value] for p in Property}
        sp = StatePoint(self._fluid, data=data)
        return sp

    def from_array(self, x, exclude=None):
        """
        Set state data from an array.

        Parameters
        ----------
        x : iterable
            Array containing property values.
        exclude : list[str], optional
            Properties to exclude. Default: []

        Raises
        ------
        ValueError if `x` does not contain the correct amount of values.

        """
        if exclude is None:
            exclude = []

        if len(x) + len(exclude) != len(Property):
            raise ValueError('Not enough values to extract.')

        count = 0
        for prop in Property:
            if prop.name not in exclude:
                self._data[prop.value] = x[count]
                count += 1

    def to_array(self, exclude=None):
        """
        Convert state data to an array.

        Parameters
        ----------
        exclude : list[str], optional
            Properties to exclude. Default: []

        Returns
        -------
        x : numpy.ndarray
            Array containing property values.

        """
        if exclude is None:
            exclude = []

        length = len(Property) - len(exclude)
        x = np.zeros(length)
        count = 0

        for prop in Property:
            if prop.name not in exclude:
                x[count] = self._data[prop.value]
                count += 1

        return x

    def properties(self, prop1=None, prop2=None, calc=None, exclude=None):
        """
        Calculate missing properties based on known ones.

        Parameters
        ----------
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

        knowns = {Property(i).name: v for i, v in enumerate(self._data) if v is not None}

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
                    self[prop] = PropsSI(
                        prop,
                        pair[0],
                        knowns[pair[0]],
                        pair[1],
                        knowns[pair[1]],
                        self._fluid
                    )

                except ValueError as e:
                    pass
