from enum import Enum


class Property(Enum):
    """
    Enumeration class defining thermodynamic properties.

    """
    T = 0  # Temperature
    P = 1  # Pressure
    D = 2  # Density
    H = 3  # Enthalpy
    S = 4  # Entropy
    Q = 5  # Quality
    Y = 6  # Fraction


class PropertyInfo:
    """
    Utility methods to retrieve information about thermodynamic properties.

    """
    _data = {
        'T': {'symbol': 'T', 'label': 'Temperature', 'unit': 'K'},
        'P': {'symbol': 'P', 'label': 'Pressure', 'unit': 'Pa'},
        'D': {'symbol': 'D', 'label': 'Density', 'unit': 'kg/m3'},
        'H': {'symbol': 'H', 'label': 'Enthalpy', 'unit': 'J/kg'},
        'S': {'symbol': 'S', 'label': 'Entropy', 'unit': 'J/kg/K'},
        'Q': {'symbol': 'Q', 'label': 'Quality', 'unit': '-'},
        'Y': {'symbol': 'Y', 'label': 'Fraction', 'unit': 'kg/m3'}
    }

    @classmethod
    def get_intkey(cls, prop):
        """
        Parameters
        ----------
        prop : Property | str | int
            Property.

        Returns
        -------
        key : int
            Integer value corresponding to the property `prop`.

        Raises
        ------
        TypeError if `prop` is not an valid property type.

        """
        if isinstance(prop, Property):
            key = prop.value
        elif isinstance(prop, str):
            key = getattr(Property, prop).value
        elif isinstance(prop, int):
            if prop > len(Property):
                raise ValueError('Property key is too large')
            key = prop
        else:
            raise TypeError('Unexpected key type')

        return key

    @classmethod
    def get_strkey(cls, prop):
        """
        Parameters
        ----------
        prop : Property | str | int
            Property.

        Returns
        -------
        key : str
            String corresponding to the property `prop`.

        Raises
        ------
        TypeError if `prop` is not an valid property type.

        """
        if isinstance(prop, Property):
            key = prop.name
        elif isinstance(prop, str):
            if prop not in cls._data.keys():
                raise ValueError('Invalid property name')
            key = prop
        elif isinstance(prop, int):
            key = Property(prop).name
        else:
            raise TypeError('Unexpected key type')

        return key

    @classmethod
    def symbol(cls, prop):
        """
        Parameters
        ----------
        prop : Property | str | int
            Property.

        Returns
        -------
        symbol : str
            Symbol corresponding to the property `prop`.

        """
        return cls._data[cls.get_strkey(prop)]['symbol']

    @classmethod
    def label(cls, prop):
        """
        Parameters
        ----------
        prop : Property | str | int
            Property.

        Returns
        -------
        label : str
            Label or description corresponding to the property `prop`.

        """
        return cls._data[cls.get_strkey(prop)]['label']

    @classmethod
    def unit(cls, prop):
        """
        Parameters
        ----------
        prop : Property | str | int
            Property.

        Returns
        -------
        unit : str
            SI unit corresponding to the property `prop`.

        """
        return cls._data[cls.get_strkey(prop)]['unit']


class PartType(Enum):
    """
    Enumeration class defining types of parts in a thermodynamic cycle.

    """
    CONDENSATOR = 0
    HEAT_SOURCE = 1
    PUMP = 2
    REHEATER_CLOSE = 3
    REHEATER_OPEN = 4
    TURBINE = 5
