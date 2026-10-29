import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.SoftmaxOptimize
open VeriTile.Triton
set_option linter.unusedVariables false

def previousMultiple : SourceFunction :=
  triton_function prev_multiple_of(a, b) in triton {
    return tl.cdiv(a, b) * b - b
  }

def callSignature : CallSignature where
  pointers := [("output_ptr",.fp32),("input_ptr",.fp32)]
  scalars := [("M",.i64),("N",.i64)]
  constexpr := ["TILE_N"]
  functions := [previousMultiple]
  readPorts := ["input_ptr"]
  writePorts := ["output_ptr"]

def body (output_ptr input_ptr : RegionName) (TILE_N : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    pid_m = tl.program_id(0).to(tl.int64)
    m = tl.full((TILE_N,), value=-float("inf"), dtype=output_ptr.dtype.element_ty)
    z = tl.full((TILE_N,), value=0, dtype=output_ptr.dtype.element_ty)
    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
        new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
        new_z = old_mass + new_mass
        m = new_m
        z = new_z
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
        new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
        new_z = old_mass + new_mass
        m = new_m
        z = new_z
    final_m = tl.max(m, 0)
    z = tl.sum(tl.exp(m - final_m) * z)
    m = final_m

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out)
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
  }

def candidateBody (output_ptr input_ptr : RegionName) (TILE_N : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    pid_m = tl.program_id(0).to(tl.int64)
    m = tl.full((TILE_N,), value=-float("inf"), dtype=output_ptr.dtype.element_ty)
    z = tl.full((TILE_N,), value=0, dtype=output_ptr.dtype.element_ty)
    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = N * pid_m + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
        new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
        new_z = old_mass + new_mass
        m = new_m
        z = new_z
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = N * pid_m + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
        new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
        new_z = old_mass + new_mass
        m = new_m
        z = new_z
    final_m = tl.max(m, 0)
    z = tl.sum(tl.exp(m - final_m) * z)
    m = final_m

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = N * pid_m + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out)
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = N * pid_m + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
  }

/-- Runtime parameter bindings are supplied outside the raw function body. -/
def kernel (output_ptr input_ptr : RegionName) (M N TILE_N : Nat) : ComputeKernel :=
  (body output_ptr input_ptr TILE_N).withScalarArguments
    [ScalarArgument.integer "N" N]

def indexCommutedKernel (Y X : RegionName) (M N B : Nat) : ComputeKernel :=
  (candidateBody Y X B).withScalarArguments
    [ScalarArgument.integer "N" N]

end VeriTile.Bench.Repairs.SoftmaxOptimize
