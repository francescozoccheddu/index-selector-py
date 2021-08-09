
class TupleView:

    def __init__(self, data):
        if not isinstance(data, tuple):
            raise TypeError("Not a tuple")
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data(index)

    def __repr__(self):
        return repr(self._data)


class QueryCosts(TupleView):

    def __init__(self, costs):
        super().__init__(costs)
        if any(not isinstance(c, (float, int)) for c in costs):
            raise TypeError("Not a float or int")
        if any(c < 0 for c in costs):
            raise ValueError("Not a non-negative number")


class Index:

    def __init__(self, fixed_cost, query_costs, size):
        if not isinstance(fixed_cost, type(QueryCosts)):
            raise TypeError(f"Not a {QueryCosts.__name__}")
        if not isinstance(fixed_cost, (float, int)) or not isinstance(size, (float, int)):
            raise TypeError("Not a float or int")
        if fixed_cost < 0 or size < 0:
            raise ValueError("Not a non-negative number")
        self._fixed_cost = fixed_cost
        self._query_costs = query_costs
        self._size = size

    @property
    def fixed_cost(self):
        return self._fixed_cost

    @property
    def query_costs(self):
        return self._query_costs

    @property
    def size(self):
        return self._size


class Model:

    def __init__(self, unindexed_query_costs, indices, max_size):
        if not isinstance(unindexed_query_costs, type(QueryCosts)):
            raise TypeError(f"Not a {QueryCosts.__name__}")
        if any(not isinstance(i, (float, int)) for i in indices):
            raise TypeError(f"Not a {Index.__name__}")
        if not isinstance(max_size, (float, int)):
            raise TypeError("Not a float or int")
        if max_size < 0:
            raise ValueError("Not a non-negative number")
        self._unindexed_query_costs = unindexed_query_costs
        self._indices = TupleView(indices)
        self._max_size = max_size

    @property
    def unindexed_query_costs(self):
        return self._unindexed_query_costs

    @property
    def indices(self):
        return self._indices

    @property
    def max_size(self):
        return self._max_size


