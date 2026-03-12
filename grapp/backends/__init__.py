__all__ = []

try:
    from .grgl import IMMUTABLE_GRG
    __all__.append("IMMUTABLE_GRG")
except ImportError:
    pass

try:
    from .spmv import SPMV_GRG
    __all__.append("SPMV_GRG")
except ImportError:
    pass

def _has_mkl():
    try:
        import ctypes.util
        mkl_lib = ctypes.util.find_library("mkl_rt")
        return mkl_lib is not None
    except Exception:
        return False

def _has_cusparse():
    try:
        import ctypes.util
        cusparse_lib = ctypes.util.find_library("cusparse")
        return cusparse_lib is not None
    except Exception:
        return False

HAS_MKL = _has_mkl()
HAS_CUSPARSE = _has_cusparse()
__all__.extend(["HAS_MKL", "HAS_CUSPARSE"])
