

from operator import index
from cplex.callbacks import UserCutCallback
from docplex.mp.callbacks.cb_mixin import *


def _ensure_positive(x):
    _ensure_type(x, (int, float))
    if x <= 0:
        raise ValueError("Not a positive number")


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


class Problem:

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


class _ModelWrapper:

    def __init__(self, problem, prune=True):
        _ensure_type(problem, Problem)
        self._problem = problem
        from docplex.mp.model import Model as CModel
        mp = CModel(name="Index Selection")
        try:
            # Decision variables
            ys = mp.binary_var_list(len(problem.indices), name="Enable index", key_format=" %s")
            uxs = mp.binary_var_list(problem.query_count, name="No index for query", key_format=" %s")
            xs = [None] * len(problem.indices)
            for i, ind in enumerate(problem.indices):
                ixs = [None] * problem.query_count
                for q, (uc, ic) in enumerate(zip(problem.unindexed_query_costs, ind.query_costs)):
                    if not prune or ic < uc:
                        ixs[q] = mp.binary_var(name=f"Index {i} for query {q}")
                xs[i] = _make_tuple(ixs)
            # Size constraint
            actual_size = mp.sum(ys[i] * ind.size for i, ind in enumerate(problem.indices))
            mp.add_constraint(actual_size <= problem.max_size, ctname="Max size")
            # Single index per query constraint
            for q in range(problem.query_count):
                actual_indices = mp.sum(xs[i][q] for i in range(len(problem.indices)) if xs[i][q] is not None) + uxs[q]
                mp.add_constraint(actual_indices == 1, ctname=f"Single index per query {q}")
            # Max index use constraint
            for i in range(len(problem.indices)):
                indices = [xs[i][q] for q in range(problem.query_count) if xs[i][q] is not None]
                actual_indices = mp.sum(indices)
                mp.add_constraint(actual_indices <= ys[i] * len(indices), ctname=f"Max uses per index {i}")
            # Target
            index_fixed_cost = mp.sum(ys[i] * ind.fixed_cost for i, ind in enumerate(problem.indices))
            indexed_query_cost = mp.sum(xs[i][q] * ind.query_costs[q] for i, ind in enumerate(problem.indices)
                                        for q in range(problem.query_count) if xs[i][q] is not None)
            unindexed_query_cost = mp.sum(uxs[q] * problem.unindexed_query_costs[q] for q in range(problem.query_count))
            mp.minimize(index_fixed_cost + indexed_query_cost + unindexed_query_cost)
        except:
            mp.end()
            raise
        self._model = mp
        self._ys = _make_tuple(ys)
        self._uxs = _make_tuple(uxs)
        self._xs = _make_tuple(xs)

    @property
    def problem(self):
        return self._problem

    @property
    def model(self):
        return self._model

    @property
    def xs(self):
        return self._xs

    @property
    def uxs(self):
        return self._uxs

    @property
    def ys(self):
        return self._ys

    def compute(self, cut=True):
        with self._model as cm:
            if cut:
                cm.register_callback(_CutCallback).set_model_wrapper(self)
            from cplex._internal._constants import CPX_MIPSEARCH_TRADITIONAL
            cm.parameters.mip.strategy.search = CPX_MIPSEARCH_TRADITIONAL
            cm.print_information()  # Remove
            s = cm.solve()
            cm.print_solution()  # Remove
            return s

disable = False

class _CutCallback(ConstraintCallbackMixin, UserCutCallback):

    def __init__(self, env):
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)

    def set_model_wrapper(self, model):
        _ensure_type(model, _ModelWrapper)
        self._mw = model

    def _add_c1(self):
        ys = self._mw.ys
        vys = self.get_values([y.index for y in ys])
        for i, ixs in enumerate(self._mw.xs):
            vixs = [self.get_values(ix.index) if ix is not None else None for ix in ixs]
            for q in range(self._mw.problem.query_count):
                if vixs[q] is not None and vixs[q] > vys[i]:
                    self.add([[ixs[q].index, ys[i].index], [1, -1]], "L", 0)

    def __call__(self):
        if not self.get_node_data():
            self.set_node_data(0)
            if not disable:
                self._add_c1()
            print(self.get_num_nodes())


def problem(unindexed_query_costs, index_query_costs, index_fixed_costs, index_sizes, max_size):
    return Problem(unindexed_query_costs, (Index(*a) for a in zip(index_fixed_costs, index_query_costs, index_sizes)), max_size)


def compute(problem, prune=True, cut=True):
    return _ModelWrapper(problem, prune).compute(cut)


def _eb_test():
    unindexed_query_costs = (6200, 2000, 800, 6700, 5000, 2000)
    query_costs = ((1300, 900, 800, 6700, 5000, 2000),
                   (6200, 700, 800, 6700, 5000, 2000),
                   (6200, 2000, 800, 1700, 2200, 2000),
                   (6200, 2000, 800, 6700, 1200, 2000),
                   (6200, 2000, 800, 2700, 4200, 750))
    fixed_costs = (200, 1200, 400, 2400, 250)
    sizes = (10, 5, 10, 8, 8)
    max_size = 19
    compute(problem(unindexed_query_costs, query_costs, fixed_costs, sizes, max_size))

def _r_test():
    compute(random_problem(50, 50, seed=1))


def random_problem(index_count, query_count, size_ratio=0.5, fixed_cost_ratio=0.2, seed=0):
    _ensure_positive(index_count)
    _ensure_positive(query_count)
    _ensure_non_negative(size_ratio)
    _ensure_non_negative(fixed_cost_ratio)
    max_min_ratio = 2
    import random
    random.seed(seed)

    def _rand():
        return random.uniform(1, max_min_ratio)

    unindexed_query_costs = [_rand() for q in range(query_count)]
    query_costs = [[_rand() for q in range(query_count)] for i in range(index_count)]
    sizes = [_rand() for i in range(index_count)]
    fixed_costs = [_rand() * fixed_cost_ratio for i in range(index_count)]
    max_size = sum(sizes) * size_ratio
    return problem(unindexed_query_costs, query_costs, fixed_costs, sizes, max_size)


if __name__ == "__main__":
    _r_test()
