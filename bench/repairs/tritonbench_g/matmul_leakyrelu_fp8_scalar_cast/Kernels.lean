import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- The shared DSL uses the method spelling of signed integer casts.
Python candidate uses tl.cast(x,tl.int64) for its nine external scalars.
On the signed representable Tensor domain, these denote the same local cast;
this quotation does not assert the original Python method call compiles.
Selected production profile: raw FP32 inputs, default TF32 dot behavior,
FP32 supplied accumulator and original activation, then explicit FP16 output.
There is one autotune configuration and both constexpr activation branches. -/
def kernel (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (ACTIVATION : Bool) : ComputeKernel :=
  triton {
    tl.region A = tl.float32, B = tl.float32, C = tl.float16
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = ($(M)).to(tl.int64) // $(128)
    if $(M) % $(128) != $(0) { num_pid_m += $(1) }
    num_pid_n = ($(N)).to(tl.int64) // $(256)
    if $(N) % $(256) != $(0) { num_pid_n += $(1) }
    active_group_m = min($(8),num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * active_group_m
    group_size_m = min(num_pid_m - first_pid_m,active_group_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    lanes_m = tl.arange(0,$(128)).to(tl.int64)
    lanes_n = tl.arange(0,$(256)).to(tl.int64)
    offs_am = (pid_m * $(128) + lanes_m) % ($(M)).to(tl.int64)
    offs_bn = (pid_n * $(256) + lanes_n) % ($(N)).to(tl.int64)
    lanes_k = tl.arange(0,$(64)).to(tl.int64)
    accumulator = tl.zeros([$(128),$(256)],dtype=tl.float32)
    k_tiles = ($(K)).to(tl.int64) // $(64)
    if $(K) % $(64) != $(0) { k_tiles += $(1) }
    for k in range($(0),k_tiles,$(1)) {
      offs_k = k.to(tl.int64) * $(64) + lanes_k
      safe_k = tl.where(offs_k < $(K),offs_k,$((0 : Int)))
      a_ptrs = A + offs_am[:,None] * ($(stride_am)).to(tl.int64) + safe_k[None,:] * ($(stride_ak)).to(tl.int64)
      b_ptrs = B + safe_k[:,None] * ($(stride_bk)).to(tl.int64) + offs_bn[None,:] * ($(stride_bn)).to(tl.int64)
      a = tl.load(a_ptrs,mask=offs_k[None,:] < $(K),other=0.0)
      b = tl.load(b_ptrs,mask=offs_k[:,None] < $(K),other=0.0)
      accumulator = tl.dot(a,b,accumulator)
    }
    if ACTIVATION {
      accumulator = tl.where(accumulator >= 0.0,accumulator,0.01 * accumulator)
    }
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * $(128) + lanes_m
    offs_cn = pid_n * $(256) + lanes_n
    safe_cm = tl.where(offs_cm < $(M),offs_cm,$((0 : Int)))
    safe_cn = tl.where(offs_cn < $(N),offs_cn,$((0 : Int)))
    c_ptrs = C + ($(stride_cm)).to(tl.int64) * safe_cm[:,None] + ($(stride_cn)).to(tl.int64) * safe_cn[None,:]
    c_mask = (offs_cm[:,None] < $(M)) & (offs_cn[None,:] < $(N))
    tl.store(c_ptrs,c,mask=c_mask)
  }

/-- Existing activation scale canonicalization, transported to the complete
selected safe-index source. Only the operand order of FP32 slope multiplication
changes; the grouped schedule, K64 live dot, casts and masked pointers remain. -/
def rightSlope (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (ACTIVATION : Bool) : ComputeKernel :=
  triton {
    tl.region A = tl.float32, B = tl.float32, C = tl.float16
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = ($(M)).to(tl.int64) // $(128)
    if $(M) % $(128) != $(0) { num_pid_m += $(1) }
    num_pid_n = ($(N)).to(tl.int64) // $(256)
    if $(N) % $(256) != $(0) { num_pid_n += $(1) }
    active_group_m = min($(8),num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * active_group_m
    group_size_m = min(num_pid_m - first_pid_m,active_group_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    lanes_m = tl.arange(0,$(128)).to(tl.int64)
    lanes_n = tl.arange(0,$(256)).to(tl.int64)
    offs_am = (pid_m * $(128) + lanes_m) % ($(M)).to(tl.int64)
    offs_bn = (pid_n * $(256) + lanes_n) % ($(N)).to(tl.int64)
    lanes_k = tl.arange(0,$(64)).to(tl.int64)
    accumulator = tl.zeros([$(128),$(256)],dtype=tl.float32)
    k_tiles = ($(K)).to(tl.int64) // $(64)
    if $(K) % $(64) != $(0) { k_tiles += $(1) }
    for k in range($(0),k_tiles,$(1)) {
      offs_k = k.to(tl.int64) * $(64) + lanes_k
      safe_k = tl.where(offs_k < $(K),offs_k,$((0 : Int)))
      a_ptrs = A + offs_am[:,None] * ($(stride_am)).to(tl.int64) + safe_k[None,:] * ($(stride_ak)).to(tl.int64)
      b_ptrs = B + safe_k[:,None] * ($(stride_bk)).to(tl.int64) + offs_bn[None,:] * ($(stride_bn)).to(tl.int64)
      a = tl.load(a_ptrs,mask=offs_k[None,:] < $(K),other=0.0)
      b = tl.load(b_ptrs,mask=offs_k[:,None] < $(K),other=0.0)
      accumulator = tl.dot(a,b,accumulator)
    }
    if ACTIVATION {
      accumulator = tl.where(accumulator >= 0.0,accumulator,accumulator * 0.01)
    }
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * $(128) + lanes_m
    offs_cn = pid_n * $(256) + lanes_n
    safe_cm = tl.where(offs_cm < $(M),offs_cm,$((0 : Int)))
    safe_cn = tl.where(offs_cn < $(N),offs_cn,$((0 : Int)))
    c_ptrs = C + ($(stride_cm)).to(tl.int64) * safe_cm[:,None] + ($(stride_cn)).to(tl.int64) * safe_cn[None,:]
    c_mask = (offs_cm[:,None] < $(M)) & (offs_cn[None,:] < $(N))
    tl.store(c_ptrs,c,mask=c_mask)
  }

/-- Semantic quotation equality does not assert Python compiler equivalence. -/
theorem shared_cast_ast (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn : Nat) (enabled : Bool) :
    kernel A B C M N K sam sak sbk sbn scm scn enabled =
      MatmulLeakyreluFp8.TypedSource.kernel A B C M N K sam sak sbk sbn scm scn enabled := rfl

def configuration : Nat × Nat × Nat × Nat × Nat × Nat := (128,256,64,8,2,8)
def activationEnabled (activation : String) : Bool := activation == "leaky_relu"
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.TypedSource
