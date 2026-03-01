from ..grg_base import GRGBase, Direction
import pygrgl

class IMMUTABLE_GRG(GRGBase):
    def __init__(self, file, load_up_edges=True):
        self._grg = pygrgl.load_immutable_grg(file, load_up_edges)
    
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
        return self._grg.is_phased
    
    @property
    def num_nodes(self):
        return self._grg.num_nodes
    
    @property
    def num_edges(self):
        return self._grg.num_edges
    
    @property
    def has_missing_data(self):
        return self._grg.has_missing_data
    
    @property
    def has_individual_ids(self):
        return self._grg.has_individual_ids
    
    @property
    def specified_bp_range(self):
        return self._grg.specified_bp_range
    
    @property
    def bp_range(self):
        return self._grg.bp_range
    
    def get_mutation_by_id(self, id: int):
        return self._grg.get_mutation_by_id(id)
    
    def get_individual_id(self, id: int):
        return self._grg.get_individual_id(id)
    
    def get_populations(self):
        return self._grg.get_populations()
    
    def get_population_id(self, id: int):
        return self._grg.get_population_id(id)
    
    def is_sample(self, id: int):
        return self._grg.is_sample(id)
    
    def get_down_edges(self, id: int):
        return self._grg.get_down_edges(id)
    
    def get_mutations_for_node(self, id: int):
        return self._grg.get_mutations_for_node(id)
    
    def get_mutation_node_miss(self):
        return self._grg.get_mutation_node_miss()
    
    def add_population(self, population):
        return self._grg.add_population(population)
    
    def set_population_id(self, population, id):
        return self._grg.set_population_id(population, id)
    
    def convert_dir(self, d: Direction):
        if d == Direction.DOWN:
            return pygrgl.TraversalDirection.DOWN
        elif d == Direction.UP:
            return pygrgl.TraversalDirection.UP

    def matmul(
        self,
        input, 
        direction,
        emit_all_nodes=False,
        by_individual=False,
        init=None,
        miss=None
    ):
        return pygrgl.matmul(self._grg, 
                             input, 
                             self.convert_dir(direction), 
                             emit_all_nodes=emit_all_nodes,
                             by_individual=by_individual,
                             init=init,
                             miss=miss)

    def shared_frontier(
        self,
        direction: Direction,
        node_pair: tuple[int, int]
    ):
        return pygrgl.shared_frontier(self._grg, 
                                       self.convert_dir(direction), 
                                       node_pair)

    def save_subset(self, 
                    file_path: str, 
                    direction: Direction,
                    sample_nodes,
                    bp_range: tuple[int, int] = (0,0)
                    ):
        return pygrgl.save_subset(self._grg, 
                                   file_path, 
                                   self.convert_dir(direction), 
                                   sample_nodes,
                                   bp_range
                                   )
    
    def save_grg(
            self,
            file_path: str
    ):
        return pygrgl.save_grg(self._grg, file_path)
