"""New native source from the independent vector-add specification (2026-10-24).
No historical Python source is claimed. Correct.lean specifies x[i]+y[i];
the existing candidate commutes the operands. All pointers are FP32.
Use Triton's actual .to cast, replacing the old Lean-only .round_to helper;
the Real projection and contextual FP equivalence are proved anew.
"""
import triton
import triton.language as tl


@triton.jit
def floatAddKernel(xReg, yReg, outReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    y = tl.load(yReg + offs)
    out = x + y
    tl.store(outReg + offs, out.to(tl.float32))


@triton.jit
def optimizedKernel(x_ptr, y_ptr, out_ptr, blockSize: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    out = y + x
    tl.store(out_ptr + offs, out.to(tl.float32))
