

def _ensure_non_negative(x):
    _ensure_type(x, (int, float))
    if x < 0:
        raise ValueError("Not a non-negative number")


def _ensure_type(x, type):
    if not isinstance(x, type):
        if not isinstance(type, tuple):
            type = (type,)
        raise TypeError("Expected" + "or ".join(t.__name__ for t in type))


def _makerepr(cls, *args, use_repr=True):
    arglist = ",".join(map(repr if use_repr else str, args))
    return f"{cls.__name__}({arglist})"


class ListView:

    @staticmethod
    def make_safe(data, validator):
        view = ListView.make(data)
        for i in view:
            validator(i)
        return view

    @staticmethod
    def make(data):
        if isinstance(data, ListView):
            return data
        if isinstance(data, tuple):
            return ListView(data)
        from collections.abc import Iterable
        if isinstance(data, Iterable):
            return ListView(tuple(data))
        raise TypeError("Not an iterable")

    @property
    def data(self):
        return self._data

    def __init__(self, data):
        if not isinstance(data, tuple):
            raise TypeError("Not a tuple")
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __repr__(self):
        return _makerepr(self.__class__, self._data)

    def __str__(self):
        return _makerepr(self.__class__, self._data, use_repr=False)


class Index:

    def __init__(self, fixed_cost, query_costs, size):
        query_costs = ListView.make_safe(query_costs, _ensure_non_negative)
        _ensure_non_negative(fixed_cost)
        _ensure_non_negative(size)
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

    def __repr__(self):
        return _makerepr(self.__class__, self._fixed_cost, self._query_costs.data, self._size)

    def __str__(self) -> str:
        return _makerepr(self.__class__, self._fixed_cost, self._query_costs.data, self._size, use_repr=False)


class Model:

    def __init__(self, unindexed_query_costs, indices, max_size):
        unindexed_query_costs = ListView.make_safe(unindexed_query_costs, _ensure_non_negative)
        indices = ListView.make_safe(indices, lambda x: _ensure_type(x, Index))
        _ensure_non_negative(max_size)
        self._unindexed_query_costs = unindexed_query_costs
        self._indices = indices
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

    def __repr__(self) -> str:
        return _makerepr(self.__class__, self._unindexed_query_costs.data, self._indices.data, self.max_size)

    def __str__(self) -> str:
        return _makerepr(self.__class__, self._unindexed_query_costs.data, self._indices.data, self.max_size, use_repr=False)


def _test():
    unindexed_query_costs = (6200, 2000, 800, 6700, 5000, 2000)
    query_costs = ((1300, 900, 800, 6700, 5000, 2000),
                   (6200, 700, 800, 6700, 5000, 2000),
                   (6200, 2000, 800, 1700, 2200, 2000),
                   (6200, 2000, 800, 6700, 1200, 2000),
                   (6200, 2000, 800, 2700, 4200, 750))
    fixed_costs = (200, 1200, 400, 2400, 250)
    sizes = (10, 5, 10, 8, 8)
    max_size = 19

    model = Model(unindexed_query_costs, (Index(*a) for a in zip(fixed_costs, query_costs, sizes)), max_size)
    print(model)


if __name__ == "__main__":
    _test()
