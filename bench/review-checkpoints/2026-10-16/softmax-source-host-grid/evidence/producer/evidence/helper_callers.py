import triton
import triton.language as tl
from source_jit_functions import next_multiple_of, prev_multiple_of
@triton.jit
def probe_next(out_ptr,a,b):
    tl.store(out_ptr,next_multiple_of(a,b))
@triton.jit
def probe_prev(out_ptr,a,b):
    tl.store(out_ptr,prev_multiple_of(a,b))
