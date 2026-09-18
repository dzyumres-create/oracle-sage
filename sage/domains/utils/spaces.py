import gym.spaces 
import numpy as np
from gym.utils import seeding
from sage.domains.utils.representations import json_to_graph

class List(gym.spaces.Space):
    """
    A json observation 
    """

    def __init__(self, dimensions, max_size=1000, dtype=np.int32):
        self.dimensions = dimensions

        self.max_size = max_size
       
        self.dtype = dtype

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self.shape = None if self.dimensions is None else tuple([max_size]*dimensions)
        self._np_random = None

    # def sample(self): #box
    #     pass

    # def contains(self, x):#box
    #     """ A method for validating x is a valid member of the Json observation.
    #     """
    #     pass

    def __eq__(self, other):  # box
        pass

class JsonGraph(gym.spaces.Box):
    """
    A json observation 
    """

    def __init__(self, converter=json_to_graph,planner=None,node_dimension=1,edge_dimension=2,width=250000):
        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self.converter = converter
        self.planner = planner
        try:
            self.shape = (1,)
        except AttributeError:
            self._shape = (1,)
        self.width = width
        # instance attribute, not a class-level constant -- different graph_convention
        # values need different widths (see taxi_env.py's GRAPH_CONVENTION_JSON_WIDTH),
        # and the default (250000) is unchanged from before this parameter existed, so
        # every existing caller that doesn't pass width= gets byte-identical behaviour.
        # Plain instance-attribute assignment to `dtype` is safe here on both gym
        # versions this codebase targets: neither gym 0.26.2 (this sandbox) nor gym
        # 0.18.0 (RCP, verified directly from its published sdist) defines `dtype` as a
        # property or uses __slots__ on Space/Box -- both their own __init__ methods set
        # self.dtype the same plain way. JsonGraph never calls super().__init__() (its
        # shape/dtype requirements are wholly different from a numeric Box's), so this
        # was already the pattern in use before this parameter was added.
        self.dtype = np.dtype(f"U{width}")
        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension



        self._np_random = None


class BinaryAction(gym.spaces.MultiDiscrete):
    """
    A binary action predicate stub
    """
    def __init__(self):
        self.nvec = np.asarray([1,1], dtype=np.int64)

class NodeAction(gym.spaces.Space):
    """
    A json observation 
    """

    def __init__(self, dimensions, dtype=np.int32):
        self.dimensions = dimensions
       
        self.dtype = dtype

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self._np_random = None


class Autoregressive(gym.spaces.Space):
    """
    A json observation 
    """

    def __init__(self, spaces, dtype=np.int32):
        self.dimensions = len(spaces)

        self.spaces = spaces
       
        #self.dtype = dtype

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        #self.shape = None if self.dimensions is None else tuple([max_size]*dimensions)
        self._np_random = None

