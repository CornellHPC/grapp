from abc import ABC, abstractmethod
from enum import Enum

class Direction(Enum):
    DOWN = 1
    UP = 2

class GRGBase(ABC):

    @property
    @abstractmethod
    def num_samples(self):
        pass

    @property
    @abstractmethod
    def num_individuals(self):
        pass

    @property
    @abstractmethod
    def num_mutations(self):
        pass

    @property
    @abstractmethod
    def ploidy(self):
        pass

    @property
    @abstractmethod
    def is_phased(self):
        pass

    @property
    @abstractmethod
    def num_nodes(self):
        pass

    @property
    @abstractmethod
    def num_edges(self):
        pass

    @property
    @abstractmethod
    def has_missing_data(self):
        pass

    @property
    @abstractmethod
    def has_individual_ids(self):
        pass

    @property
    @abstractmethod
    def specified_bp_range(self):
        pass

    @property
    @abstractmethod
    def bp_range(self):
        pass

    @abstractmethod
    def get_mutation_by_id(self, id: int):
        pass

    @abstractmethod
    def get_individual_id(self, id: int):
        pass

    @abstractmethod
    def get_populations(self):
        pass

    @abstractmethod
    def get_population_id(self, id: int):
        pass

    @abstractmethod
    def is_sample(self, id: int):
        pass

    @abstractmethod
    def get_down_edges(self, id: int):
        pass

    @abstractmethod
    def get_mutations_for_node(self, id: int):
        pass

    @abstractmethod
    def get_mutation_node_miss(self):
        pass

    @abstractmethod
    def add_population(self, population: str):
        pass

    @abstractmethod
    def set_population_id(self, population: str, id: int):
        pass

    @abstractmethod
    def matmul(
        self,
        input,
        direction: Direction,
        emit_all_nodes: bool,
        by_individual: bool,
        init,
        miss
    ):
        pass

    @abstractmethod
    def shared_frontier(
        self,
        direction: Direction,
        node_pair: tuple[int, int]
    ):
        pass

    @abstractmethod
    def save_subset(self, 
                    file_path: str, 
                    direction: Direction,
                    sample_nodes,
                    bp_range: tuple[int, int]
                    ):
        pass

    @abstractmethod
    def save_grg(self,
                 file_path: str
                 ):
        pass
