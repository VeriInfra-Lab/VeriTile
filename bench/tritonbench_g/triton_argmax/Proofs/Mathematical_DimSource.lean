import bench.tritonbench_g.triton_argmax.Proofs.Mathematical_TypedSource
namespace VeriTile.Bench.TritonBenchG.TritonArgmax.TypedSource
open VeriTile.Triton Lean
syntax "argmaxDim%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(argmaxDim%[$dtype:tritonDType]) =>
    `(fun (inp : RegionName) (out_index : Region .int)
      (M N K BLOCK_M BLOCK_N : Nat) (INT64_INDEX : Bool) =>
      ComputeKernel.mk [inp] [Region.cast out_index] (triton {
        tl.region inp = $dtype
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        if INT64_INDEX {
          pid_m = (pid_m).to(tl.int64)
          pid_k = (pid_k).to(tl.int64)
        }
        m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
        max_values = tl.full([$(BLOCK_M)], dtype=tl.float32, value=float("-inf"))
        argmax_values = tl.full([$(BLOCK_M)], dtype=tl.int64, value=0)
        for start_n in range($(0), $(N), $(BLOCK_N)) {
          n_offset = start_n + tl.arange(0, $(BLOCK_N))
          offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
          mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
          inp_ptrs = inp + offset
          inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
          local_max, local_argmax := tl.max(inp_vals, 1,
            return_indices=True, return_indices_tie_break_left=True)
          update = local_max.to(tl.float32) > max_values
          max_values = tl.where(update, local_max.to(tl.float32), max_values)
          argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
        }
        offset_index = m_offset * $(K) + pid_k
        out_index_ptrs = out_index + offset_index
        mask1 = m_offset < $(M)
        tl.store(out_index_ptrs, argmax_values, mask=mask1)
      }).surfaceBody)

def dim (storage : Storage) : RegionName → Region .int → Nat → Nat → Nat → Nat → Nat → Bool → ComputeKernel :=
  match storage with
  | .fp16 => argmaxDim%[tl.float16]
  | .fp32 => argmaxDim%[tl.float32]
end VeriTile.Bench.TritonBenchG.TritonArgmax.TypedSource
