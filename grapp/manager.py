"""
Backend manager for handling GPU and CPU computation backends in grapp.
"""
import numpy as np
import pygrgl
from typing import Optional, Literal

BackendType = Literal["cpu", "gpu"]

class BackendManager:
    """
    Manages the computation backend (CPU or GPU) for GRG operations.
    """
    
    def __init__(self, backend: BackendType = "cpu"):
        """
        Initialize the backend manager.
        
        :param backend: Either "cpu" or "gpu"
        :param gpu_grg: Pre-loaded GPUGRG object if using GPU backend
        """
        self._backend = backend
    
    @property
    def backend(self) -> BackendType:
        return self._backend
    
    def set_backend(self, backend: BackendType):
        """Switch backend."""
        self._backend = backend

    
    def matmul(
        self,
        grg,
        matrix: np.ndarray,
        direction: pygrgl.TraversalDirection,
        emit_all_nodes: bool = False,
        by_individual: bool=False,
        init=None,
    ) -> np.ndarray:
        """
        Perform matrix multiplication using the selected backend.
        
        :param grg: The GRG object
        :param matrix: Input matrix
        :param direction: Traversal direction (UP or DOWN)
        :param init: Initialization mode ("zero", "xtx", "vector", "matrix")
        :return: Result matrix
        """
        if self._backend == "gpu":
            if emit_all_nodes:
                assert False, "emit_all_nodes=True is not supported on GPU backend"
            return grg.matmul(matrix, direction, by_individual, init)
        else:
            return pygrgl.matmul(grg, matrix, direction, emit_all_nodes, by_individual, init)

    def dot_product(
        self,
        grg,
        vector: np.ndarray,
        direction: pygrgl.TraversalDirection,
    ) -> np.ndarray:
        """
        Perform dot product using the selected backend.
        
        :param grg: The GRG object
        :param vector: Input vector
        :param direction: Traversal direction (UP or DOWN)
        :return: Result vector
        """
        if self._backend == "gpu":
            return grg.dot_product(vector, direction)
        else:
            return pygrgl.dot_product(grg, vector, direction)


# Global backend manager instance
_global_backend_manager: Optional[BackendManager] = None

def get_backend_manager() -> BackendManager:
    """Get the global backend manager instance."""
    global _global_backend_manager
    if _global_backend_manager is None:
        _global_backend_manager = BackendManager(backend="cpu")
    return _global_backend_manager

def set_backend(backend: BackendType):
    """Set the global backend."""
    global _global_backend_manager
    if _global_backend_manager is None:
        _global_backend_manager = BackendManager(backend=backend)
    else:
        _global_backend_manager.set_backend(backend)