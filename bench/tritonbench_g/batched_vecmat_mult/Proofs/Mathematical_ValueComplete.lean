import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_ValueExecution

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] FP.add FP.mul treeValue

theorem fold_write_memory (p : Parameters) (initial s : State Word)
    (values : TileIndex [p.BM,p.BN] → Word) (xs : List (TileIndex [p.BM,p.BN])) :
    (xs.foldl (fun t i => t.write p.output (outputOffset p initial i) (.mk .real (values i))) s).mem =
      xs.foldl (fun mem i => fun r o => if r = p.output ∧ o = outputOffset p initial i then
        .mk .real (values i) else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons, ih]
    rfl

#axiomsClean fold_write_memory
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
