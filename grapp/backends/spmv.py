from ..grg_base import GRGBase, Direction
import numpy as np
import pygrgl
from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.mkl import MklPlan
from pygrgl_spmv.backends.cusparse import CusparsePlan


class SPMV_GRG(GRGBase):
    def __init__(self, file, op, load_up_edges=True):
        self._grg = pygrgl.load_immutable_grg(file, load_up_edges)
        self._op = op
    
    @property
    def num_samples(self):
        return self._grg.num_samples
    
    @property
    def num_individuals(self):
        return self._grg.num_individuals
    
    @property
    def num_mutations(self):
        return self._grg.num_mutations
    
    @property
    def ploidy(self):
        return self._grg.ploidy
    
    @property
    def is_phased(self):
        return NotImplementedError
    
    @property
    def num_nodes(self):
        return self._grg.num_nodes
    
    @property
    def num_edges(self):
        return self._grg.num_edges
    
    @property
    def has_missing_data(self):
        # TODO
        return self._grg.has_missing_data
    
    @property
    def has_individual_ids(self):
        raise NotImplementedError
    
    @property
    def specified_bp_range(self):
        raise NotImplementedError
    
    @property
    def bp_range(self):
        raise NotImplementedError
    
    def get_mutation_by_id(self, id: int):
        # TODO
        return self._grg.get_mutation_by_id(id)
    
    def get_individual_id(self, id: int):
        raise NotImplementedError
    
    def get_populations(self):
        raise NotImplementedError
    
    def get_population_id(self, id: int):
        raise NotImplementedError
    
    def is_sample(self, id: int):
        raise NotImplementedError
    
    def get_down_edges(self, id: int):
        raise NotImplementedError
    
    def get_mutations_for_node(self, id: int):
        raise NotImplementedError
    
    def get_mutation_node_miss(self):
        raise NotImplementedError
    
    def add_population(self, population):
        raise NotImplementedError
    
    def set_population_id(self, population, id):
        raise NotImplementedError
    
    def convert_dir(self, d: Direction):
        if d == Direction.DOWN:
            return "down"
        elif d == Direction.UP:
            return "up"

    def matmul(
        self,
        input, 
        direction,
        emit_all_nodes=False,
        by_individual=False,
        init=None,
        miss=None
    ):
        assert not emit_all_nodes, "emit_all_nodes is not supported"
        return self._op.matmul(
            input,
            self.convert_dir(direction),
            by_individual=by_individual,
            init=init,
            miss=miss
        )

    def shared_frontier(
        self,
        direction: Direction,
        node_pair: tuple[int, int]
    ):
        raise NotImplementedError

    def save_subset(self, 
                    file_path: str, 
                    direction: Direction,
                    sample_nodes,
                    bp_range: tuple[int, int] = (0,0)
                    ):
        raise NotImplementedError

    def save_grg(
            self,
            file_path: str
    ):
        raise NotImplementedError

class SPMV_GRG_MKL(SPMV_GRG):
    def __init__(self, file, load_up_edges=True, nthreads=64):
        op = SpmvGRG(
            file,
            {
                "type": "mkl",
                "plan_up": MklPlan.from_any({"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": nthreads}),
                "plan_down": MklPlan.from_any({"k_hint": None, "store": "T", "fmt": "CSC", "n_threads": nthreads}),
            },
            np.float64,
            np.uintp,
            cache_dir="pygrgl_spmv_cache",
        )
        super().__init__(file, op=op, load_up_edges=load_up_edges)

class SPMV_GRG_cuSparse(SPMV_GRG):
    def __init__(self, file, load_up_edges=True, k_hint=1):
        op = SpmvGRG(
            file,
            {
                "type": "cusparse",
                "plan_up": CusparsePlan.from_any({
                    "k_hint": k_hint, 
                    "store": "N", 
                    "fmt": "CSR", 
                    "opA": "N",
                    "opB": "N",
                    "orderB": "ROW",
                    "orderC": "ROW",
                    "algo": "DEFAULT",
                }),
                "plan_down": CusparsePlan.from_any({
                    "k_hint": k_hint,
                    "store": "N",
                    "fmt": "CSR",
                    "opA": "T",
                    "opB": "N",
                    "orderB": "ROW",
                    "orderC": "ROW",
                    "algo": "DEFAULT",
                }),
            },
            np.float64,
            np.uintp,
            cache_dir="pygrgl_spmv_cache",
        )
        super().__init__(file, op=op, load_up_edges=load_up_edges)
