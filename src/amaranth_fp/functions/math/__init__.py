"""Mathematical function approximations using polynomial evaluation."""
from .fp_exp2 import FPExp2
from .fp_log2 import FPLog2
from .fp_log10 import FPLog10
from .fp_exp10 import FPExp10
from .fp_asin import FPAsin
from .fp_acos import FPAcos
from .fp_atan import FPAtan
from .fp_sinh import FPSinh
from .fp_cosh import FPCosh
from .fp_tanh import FPTanh
from .fp_asinh import FPAsinh
from .fp_acosh import FPAcosh
from .fp_atanh import FPAtanh
from .fp_erf import FPErf
from .fp_erfc import FPErfc
from .fp_cbrt import FPCbrt
from .fp_reciprocal import FPReciprocal
from .fp_rsqrt import FPRsqrt

__all__ = [
    "FPExp2", "FPLog2", "FPLog10", "FPExp10",
    "FPAsin", "FPAcos", "FPAtan",
    "FPSinh", "FPCosh", "FPTanh",
    "FPAsinh", "FPAcosh", "FPAtanh",
    "FPErf", "FPErfc",
    "FPCbrt", "FPReciprocal", "FPRsqrt",
]
