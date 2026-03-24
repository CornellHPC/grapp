from ..grg_base import GRGBase, Direction
import os
import numpy as np
import pygrgl
from pygrgl_spmv import SpmvGRG
from pygrgl_spmv.backends.mkl import MklBackend, MklPlan, MklPlanPair
from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair, CusparseBackend


class SPMV_GRG(GRGBase):
    def __init__(self, file, backend, dtype, index_dtype):
        if (file.endswith(".grg")):
            print("Loading a GRG file instead of grg_spmv file. Converting will be slow.")
            op = SpmvGRG(file, backend, dtype, index_dtype)
            print(f"Converted dataset is saved at pygrgl_spmv_artifacts")
        elif (file.endswith(".grg_spmv")):
            op = SpmvGRG(file, backend, dtype, index_dtype)
        else:
            assert False, "Unsupported file format. Only .grg and .grg_spmv are supported."

        self._op = op
    
    @property
    def num_samples(self):
        return self._op.num_samples

    @property
    def num_individuals(self):
        return self._op.num_individuals

    @property
    def num_mutations(self):
        return self._op.num_mutations

    @property
    def ploidy(self):
        return self._op.ploidy

    @property
    def is_phased(self):
        return NotImplementedError
    
    @property
    def num_nodes(self):
        return self._op.num_nodes
    
    @property
    def num_edges(self):
        return self._op.num_edges
    
    @property
    def has_missing_data(self):
        return self._op.has_missing_data
    
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
        return self._op.get_mutation_by_id(id)
    
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
        return self._op.matmul(
            input,
            self.convert_dir(direction),
            emit_all_nodes=emit_all_nodes,
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
        print("Please use convert_grg function to generate .grg_spmv_file")
        raise NotImplementedError

    @staticmethod
    def convert_from_grg(
            src_path: str,
            dst_path: str
    ):
        if not src_path.endswith(".grg"):
            raise ValueError("Source file must be a .grg file")
        try:
            dummy_pair = MklPlanPair.from_dicts(
                {"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": 1},
                {"k_hint": None, "store": "T", "fmt": "CSC", "n_threads": 1},
            )
            dummy_backend = MklBackend(pair=dummy_pair, instrumentation=False, log_level="WARNING")
            _op = SpmvGRG(src_path, dummy_backend, np.float64, np.int32, dst_path)
        except Exception as e:
            try:
                print(f"MKL backend unavailable ({e}), falling back to cuSPARSE.")
                dummy_pair = CusparsePlanPair.from_dicts(
                    {"k_hint": None, "store": "N", "fmt": "CSR", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT"},
                    {"k_hint": None, "store": "N", "fmt": "CSR", "opA": "T", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT"},
                )
                dummy_backend = CusparseBackend(pair=dummy_pair, instrumentation=False, log_level="WARNING")
                _op = SpmvGRG(src_path, dummy_backend, np.float64, np.int32, dst_path)
            except Exception as e:
                assert False, "No backend available."

        name = os.path.splitext(os.path.basename(src_path))[0]
        resolved = os.path.realpath(os.path.expanduser(src_path))
        saved_path = os.path.join(dst_path, "_abs", resolved.lstrip("/").replace(".grg", ".grg_spmv"))
        final_path = os.path.join(dst_path, name + ".grg_spmv")
        os.rename(saved_path, final_path)

        print(f"Constructed file saved at {final_path}")


class SPMV_GRG_MKL(SPMV_GRG):
    def __init__(self, file, nthreads=64):
        backend = MklBackend(
            pair=MklPlanPair(
                plan_up=MklPlan.from_dict({"k_hint": None, "store": "N", "fmt": "CSR", "n_threads": nthreads}),
                plan_down=MklPlan.from_dict({"k_hint": None, "store": "T", "fmt": "CSC", "n_threads": nthreads}),
            ),
            instrumentation=False,
            log_level="WARNING",
        )
        super().__init__(file, backend, np.float64, np.int32)

class SPMV_GRG_cuSparse(SPMV_GRG):
    def __init__(self, file, k_hint=1):
        backend = CusparseBackend(
            pair=CusparsePlanPair.from_dicts(
                {"k_hint": k_hint, "store": "N", "fmt": "CSR", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT"},
                {"k_hint": k_hint, "store": "N", "fmt": "CSR", "opA": "T", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT"},
            ),
            instrumentation=False,
            log_level="WARNING",
        )
        super().__init__(file, backend, np.float64, np.int32)
