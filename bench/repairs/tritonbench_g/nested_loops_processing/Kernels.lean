import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Actual selected source, including the initial runtime integer assignments.
The explicitly new Python candidate only mirrors the existing Lean index product. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Kernels
open VeriTile.Triton

def sourceSignature : CallSignature where
  pointers := [("in_ptr",.i32),("out_ptr",.i32)]
  scalars := [("stride_m",.integer),("stride_n",.integer),("n_elements",.integer)]
  constexpr := []
  readPorts := ["in_ptr"]
  writePorts := ["out_ptr"]

/-- Complete selected original body. -/
def quotedOriginal (in_ptr out_ptr : RegionName) (stride_m stride_n n_elements : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args [stride_m := stride_m, stride_n := stride_n, n_elements := n_elements] in
  triton {
    stride_m = tl.cast(stride_m, tl.int64)
    stride_n = tl.cast(stride_n, tl.int64)
    n_elements = tl.cast(n_elements, tl.int64)
    offs_am = tl.arange(0, 2).to(tl.int64)
    offs_an = tl.arange(0, 2).to(tl.int64)
    a_offsets = offs_am[:, None] * stride_m + offs_an[None, :] * stride_n

    offs_cm = tl.arange(0, 2).to(tl.int64)
    offs_cn = tl.arange(0, 2).to(tl.int64)
    c_offsets = stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]
    # A contiguous one-column input duplicates lanes (0, 1) and (1, 0).
    # Their cached input words agree; retain only the earlier physical store.
    unique = (offs_cm[:, None] == 0) | (stride_m > 1) | (offs_cn[None, :] == 1)

    for i in range(0, 2):
        a1 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                     mask=a_offsets < n_elements, other=0)
        for j in range(0, 2):
            a_offsets += 2 * stride_n
            a2 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                         mask=a_offsets < n_elements, other=0)
            for k in range(0, 2):
                a_offsets += 2 * stride_n
                a3 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                             mask=a_offsets < n_elements, other=0)
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a1,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a2,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a3,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
        a_offsets += 2 * stride_n
  }

/-- Complete existing candidate with a clearly authored Python counterpart body. -/
def quotedCommuted (in_ptr out_ptr : RegionName) (stride_m stride_n n_elements : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args [stride_m := stride_m, stride_n := stride_n, n_elements := n_elements] in
  triton {
    stride_m = tl.cast(stride_m, tl.int64)
    stride_n = tl.cast(stride_n, tl.int64)
    n_elements = tl.cast(n_elements, tl.int64)
    offs_am = tl.arange(0, 2).to(tl.int64)
    offs_an = tl.arange(0, 2).to(tl.int64)
    a_offsets = offs_am[:, None] * stride_m + stride_n * offs_an[None, :]

    offs_cm = tl.arange(0, 2).to(tl.int64)
    offs_cn = tl.arange(0, 2).to(tl.int64)
    c_offsets = stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]
    # A contiguous one-column input duplicates lanes (0, 1) and (1, 0).
    # Their cached input words agree; retain only the earlier physical store.
    unique = (offs_cm[:, None] == 0) | (stride_m > 1) | (offs_cn[None, :] == 1)

    for i in range(0, 2):
        a1 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                     mask=a_offsets < n_elements, other=0)
        for j in range(0, 2):
            a_offsets += 2 * stride_n
            a2 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                         mask=a_offsets < n_elements, other=0)
            for k in range(0, 2):
                a_offsets += 2 * stride_n
                a3 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                             mask=a_offsets < n_elements, other=0)
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a1,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a2,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a3,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
        a_offsets += 2 * stride_n
  }

/-- The retained contiguous caller scope supplies the actual unit minor stride. -/
abbrev originalKernel (in_ptr out_ptr : RegionName) (count cols : Nat) : ComputeKernel :=
  quotedOriginal in_ptr out_ptr cols 1 count
abbrev commutedKernel (in_ptr out_ptr : RegionName) (count cols : Nat) : ComputeKernel :=
  quotedCommuted in_ptr out_ptr cols 1 count

def kernel (candidate : Bool) : RegionName → RegionName → Nat → Nat → ComputeKernel :=
  if candidate then commutedKernel else originalKernel

def source (count cols : Nat) : ComputeKernel := kernel Bool.false "input" "output" count cols
def commuted (count cols : Nat) : ComputeKernel := kernel Bool.true "input" "output" count cols
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Kernels
