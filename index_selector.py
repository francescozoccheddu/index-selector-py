

def _ensure_non_negative(x):
    _ensure_type(x, (int, float))
    if x < 0:
        raise ValueError("Not a non-negative number")


def _ensure_non_empty(x):
    if len(x) == 0:
        raise ValueError("Empty")


def _ensure_type(x, type):
    if not isinstance(x, type):
        if not isinstance(type, tuple):
            type = (type,)
        raise TypeError("Expected" + "or ".join(t.__name__ for t in type))


def _make_repr(cls, *args, use_repr=True):
    arglist = ",".join(map(repr if use_repr else str, args))
    return f"{cls.__name__}({arglist})"


def _make_tuple(data, validator=None):
    if not isinstance(data, tuple):
        from collections.abc import Iterable
        _ensure_type(data, Iterable)
        data = tuple(data)
    if validator is not None:
        for i in data:
            validator(i)
    return data


class Index:

    def __init__(self, fixed_cost, query_costs, size):
        query_costs = _make_tuple(query_costs, _ensure_non_negative)
        _ensure_non_empty(query_costs)
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
        return _make_repr(self.__class__, self._fixed_cost, self._query_costs, self._size)

    def __str__(self) -> str:
        return _make_repr(self.__class__, self._fixed_cost, self._query_costs, self._size, use_repr=False)


class Model:

    def __init__(self, unindexed_query_costs, indices, max_size):
        unindexed_query_costs = _make_tuple(unindexed_query_costs, _ensure_non_negative)
        _ensure_non_empty(unindexed_query_costs)
        query_count = len(unindexed_query_costs)
        indices = _make_tuple(indices, lambda x: _ensure_type(x, Index))
        for i in indices:
            if len(i.query_costs) != query_count:
                raise ValueError("Query count does not match")
        _ensure_non_negative(max_size)
        self._unindexed_query_costs = unindexed_query_costs
        self._indices = indices
        self._max_size = max_size

    @property
    def query_count(self):
        return len(self._unindexed_query_costs)

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
        return _make_repr(self.__class__, self._unindexed_query_costs, self._indices, self.max_size)

    def __str__(self) -> str:
        return _make_repr(self.__class__, self._unindexed_query_costs, self._indices, self.max_size, use_repr=False)


class OptimizationModel:

    def __init__(self, model, prune=True):
        _ensure_type(model, Model)
        self._source = model
        from docplex.mp.model import Model as CModel
        mp = CModel("Index Selection")
        try:
            # Decision variables
            ys = mp.binary_var_list(len(model.indices), name="yi")
            uxs = mp.binary_var_list(len(model.indices), name="uxq")
            xs = [None] * len(model.indices)
            for ii, i in enumerate(model.indices):
                ixs = [None] * model.query_count
                for q, (uc, ic) in enumerate(zip(model.unindexed_query_costs, i.query_costs)):
                    if not prune or ic < uc:
                        ixs[q] = mp.binary_var(name=f"xiq_{ii}_{q}")
                xs[ii] = _make_tuple(ixs)
            # Size constraint
            # Single index per query constraint
            # Max index use constraint
        except:
            mp.end()
        self._mp = mp
        self._ys = _make_tuple(ys)
        self._uxs = _make_tuple(uxs)
        self._xs = _make_tuple(xs)

    @property
    def source(self):
        return self._source

    @property
    def model(self):
        return self._mp

    @property
    def xs(self):
        return self._xs

    @property
    def uxs(self):
        return self._uxs

    def ys(self):
        return self._ys


def _compute_cuts(solution):
    pass


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
    om = OptimizationModel(model)
    pass


if __name__ == "__main__":
    _test()
