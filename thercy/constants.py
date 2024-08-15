from enum import Enum


class Property(Enum):
    """
    Enumeration representing thermodynamic properties.

    The enumeration values and their corresponding properties are as follows:
    - T (0): Temperature
    - P (1): Pressure
    - D (2): Density
    - H (3): Enthalpy
    - S (4): Entropy
    - Q (5): Quality
    - Y (6): Fraction
    """
    T = 0
    P = 1
    D = 2
    H = 3
    S = 4
    Q = 5
    Y = 6


class PropertyInfo:
    """
    Informations about thermodynamic properties.
    """
    _data = {
        'T': {'symbol': 'T', 'label': 'Temperature', 'unit': 'K'},
        'P': {'symbol': 'P', 'label': 'Pressure', 'unit': 'Pa'},
        'D': {'symbol': 'D', 'label': 'Density', 'unit': 'kg/m3'},
        'H': {'symbol': 'H', 'label': 'Enthalpy', 'unit': 'J/kg'},
        'S': {'symbol': 'S', 'label': 'Entropy', 'unit': 'J/kg/K'},
        'Q': {'symbol': 'Q', 'label': 'Quality', 'unit': '-'},
        'Y': {'symbol': 'Y', 'label': 'Fraction', 'unit': 'kg/s'}
    }

    @classmethod
    def get_intkey(cls, prop):
        """
        Returns the integer key associated with the property ``prop``.

        Parameters
        ----------
        prop : Property | str | int
            The parameter representing the key for Property.

        Returns
        -------
        int
            The integer key associated with the given property.

        Raises
        ------
        ValueError
            If the property key is invalid.
        TypeError
            If the type of key is unexpected.
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
        Returns the string key associated with the property ``prop``.

        Parameters
        ----------
        prop : Property | str | int
            The parameter representing the key for Property.

        Returns
        -------
        str
            The string key associated with the given property.

        Raises
        ------
        ValueError
            If the property key is invalid.
        TypeError
            If the type of key is unexpected.
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
        Returns the symbol associated with the property ``prop``.

        Parameters
        ----------
        prop : Property | str | int
            The parameter representing the key for Property.

        Returns
        -------
        str
            Symbol corresponding to the given property.
        """
        return cls._data[cls.get_strkey(prop)]['symbol']

    @classmethod
    def label(cls, prop):
        """
        Returns the label or description associated with the property ``prop``.

        Parameters
        ----------
        prop : Property | str | int
            The parameter representing the key for Property.

        Returns
        -------
        str
            Label or description corresponding to the given property.
        """
        return cls._data[cls.get_strkey(prop)]['label']

    @classmethod
    def unit(cls, prop):
        """
        Returns the SI unit associated with the property ``prop``.

        Parameters
        ----------
        prop : Property | str | int
            The parameter representing the key for Property.

        Returns
        -------
        str
            SI unit corresponding to the given property.
        """
        return cls._data[cls.get_strkey(prop)]['unit']


class PartType(Enum):
    """
    Enumeration representing different types of parts in a thermodynamic system.

    The enumeration values and their corresponding properties are as follows:
    - CONDENSATOR    (0): Condensator
    - EVAPORATOR     (1): Evaporator
    - HEAT_EXCHANGER (2): Heat exchanger
    - HEAT_SOURCE    (3): Heat source
    - PUMP           (4): Pump
    - REHEATER_CLOSE (5): Closed reheater
    - REHEATER_OPEN  (6): Open reheater
    - TURBINE        (7): Turbine
    """
    CONDENSATOR = 0
    EVAPORATOR = 1
    HEAT_EXCHANGER = 2
    HEAT_SOURCE = 3
    PUMP = 4
    REHEATER_CLOSED = 5
    REHEATER_OPEN = 6
    TURBINE = 7
