import numpy as np

from thercy.utils import list_like
from .cycle import StateCycle
from .point import StatePoint


class StateGraph:
    """
    Graph representing a thermodynamic state system composed of
    interconnected parts.

    Properties
    ----------
    fluid : str
        Fluid composition or mixture name.
    states : StateCycle
        State cycle containing state points corresponding to edges.

    """
    def __init__(self, fluid):
        """
        Initialize a StateGraph instance.

        Parameters
        ----------
        fluid : str
            Fluid composition or mixture name.

        """
        self._nodes: dict[str, any] = {}
        self._edges: list[(str, str)] = []
        self._edges_map: dict[(str, str): int] = {}

        self.states: StateCycle = StateCycle(fluid)

    def __iter__(self):
        """Iterator over graph nodes."""
        return iter(self._nodes.values())

    def __len__(self):
        """Get the number of nodes in the graph."""
        return len(self._nodes)

    def __str__(self):
        """String representation of the state cycle."""
        return str(self.states)

    def __getitem__(self, key):
        """
        Get a node by key.

        Parameters
        ----------
        key : str
            Node key.

        Returns
        -------
        any
            Node corresponding to the key.

        """
        return self._nodes[key]

    @property
    def points(self):
        """Get the number of edges in the graph."""
        return len(self._edges)

    @property
    def nodes(self):
        """Get the nodes in the graph."""
        return self._nodes

    @property
    def edges(self):
        """Get the edges in the graph."""
        return self._edges

    def get_edge_index(self, edge):
        """
        Get the index of an edge.

        Parameters
        ----------
        edge : tuple
            Edge to find.

        Returns
        -------
        int
            Index of the edge.

        """
        return self._edges_map[edge]

    def get_state(self, edge):
        return self.states[self.get_edge_index(edge)]

    def get_outlets(self, key, inlet_key=None, index=False):
        """
        Get outlet keys connected to a node.

        Parameters
        ----------
        key : str
            Key of the node.
        inlet_key : str, optional
            Inlet key to filter outlets. Default: None
        index : bool, optional
            If True, return indices instead of keys. Default: False

        Returns
        -------
        list[str | int]
            List of outlet keys connected to the node.

        """
        if inlet_key is None:
            return {i if index else edge[1] for i, edge in enumerate(self._edges) if edge[0] == key}

        if index:
            raise NotImplementedError()

        outlets = []
        for conn in self._nodes[key].connections:
            for inlet in conn.inlets:
                if inlet.label == inlet_key:
                    for outlet in conn.outlets:
                        outlets.append(outlet.label)
                    break

        return outlets

    def add_part(self, part):
        """
        Add a part to the graph.

        Parameters
        ----------
        part : BasePart
            Part to add.

        Raises
        ------
        ValueError
            If the part already exists in the graph.

        """
        if part.label in self._nodes:
            raise ValueError('part already exists')

        self._nodes[part.label] = part

        for conn in part.connections:
            for inlet in conn.inlets:
                edge = (inlet.label, part.label)
                edge_index = len(self._edges)
                if edge not in self._edges:
                    self._edges.append(edge)
                    self._edges_map[edge] = edge_index
                    self.states[edge_index] = StatePoint(self.states.fluid)

            for outlet in conn.outlets:
                edge = (part.label, outlet.label)
                edge_index = len(self._edges)
                if edge not in self._edges:
                    self._edges.append(edge)
                    self._edges_map[edge] = edge_index
                    self.states[edge_index] = StatePoint(self.states.fluid)

    def cloud_points(self, n=50, precise=False):
        """
        Generate cloud points between consecutive state points.

        Parameters
        ----------
        n : int, optional
            Number of points between each pair of state points.
            Default: 50
        precise : bool, optional
            Use a more precise version of interpolation by enthalpy
            instead of by temperature, slower. Default: False

        Returns
        -------
        list[StateCycle]
            List of state cycles containing the cloud points.

        """
        cloud = [StateCycle(self.states.fluid) for _ in range(self.points)]

        for i, edge1 in enumerate(self.edges):
            part_label = edge1[1]

            index1 = self.get_edge_index(edge1)
            state1 = self.states[index1]

            for outlet in self._nodes[part_label].get_outlets(edge1[0]):
                edge2 = (part_label, outlet.label)
                index2 = self.get_edge_index(edge2)
                state2 = self.states[index2]

                prop_x = next((prop.name for prop in self.states.constant_properties(index1, index2) if
                               prop.name not in ('S', 'Q', 'Y')), 'H' if precise else 'T')

                s_diff = state2['S'] - state1['S']
                x_diff = state2[prop_x] - state1[prop_x]

                s = np.linspace(state1['S'], state2['S'], n - 1)
                s = np.append(s, state2['S'])

                x = (x_diff / s_diff) * (s - state1['S']) + state1[prop_x] if abs(s_diff) > 1e-7 \
                    else np.linspace(state1[prop_x], state2[prop_x], n)

                for j in range(n):
                    cloud[i][j, 'S'] = s[j]
                    cloud[i][j, prop_x] = x[j]
                    cloud[i][j].properties(calc=['T'])
                    cloud[i][j].properties('T', 'S', exclude=['T'])
                    cloud[i][j, 'Y'] = state1['Y']

        return cloud
