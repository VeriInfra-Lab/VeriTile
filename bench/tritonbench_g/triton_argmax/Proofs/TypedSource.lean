import bench.tritonbench_g.matrix_reduction.Proofs.TypedSource

namespace VeriTile.Bench.TritonBenchG.TritonArgmax.TypedSource
open VeriTile.Triton Lean
abbrev Storage := MatrixReduction.TypedSource.Storage

syntax "argmaxStage1%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(argmaxStage1%[$dtype:tritonDType]) =>
    `(fun (inp mid_value : RegionName) (mid_index : Region .int)
      (M BLOCK_SIZE : Nat) (INT64_INDEX : Bool) =>
      ComputeKernel.mk [inp] [mid_value, Region.cast mid_index] (triton {
        tl.region inp = $dtype, mid_value = $dtype
        pid = tl.program_id(0)
        if INT64_INDEX { pid = (pid).to(tl.int64) }
        offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        inp_ptrs = inp + offset
        mask = offset < $(M)
        inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
        max_index = max_index + pid * $(BLOCK_SIZE)
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val)
        tl.store(max_index_ptr, max_index.to(tl.int64))
      }).surfaceBody)

def stage1 (storage : Storage) : RegionName → RegionName → Region .int → Nat → Nat → Bool → ComputeKernel :=
  match storage with
  | .fp16 => argmaxStage1%[tl.float16]
  | .fp32 => argmaxStage1%[tl.float32]

syntax "argmaxStage2%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(argmaxStage2%[$dtype:tritonDType]) =>
    `(fun (mid_value : RegionName) (mid_index out : Region .int) (mid_size BLOCK_MID : Nat) =>
      ComputeKernel.mk [mid_value, Region.cast mid_index] [Region.cast out] (triton {
        tl.region mid_value = $dtype
        offset = tl.arange(0, $(BLOCK_MID))
        mid_ptrs = mid_value + offset
        mask = offset < $(mid_size)
        mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
        index_val = tl.argmax(mid_val, axis=0)
        mid_index_ptrs = mid_index + index_val
        out_val = tl.load(mid_index_ptrs)
        tl.store(out, out_val)
      }).surfaceBody)

def stage2 (storage : Storage) : RegionName → Region .int → Region .int → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => argmaxStage2%[tl.float16]
  | .fp32 => argmaxStage2%[tl.float32]
end VeriTile.Bench.TritonBenchG.TritonArgmax.TypedSource
