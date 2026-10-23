import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.RawKernel
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Native

namespace VeriTile.Bench.Repairs.LogsumexpFwd.RawNative
open VeriTile.Triton FP.Structural Native
noncomputable section

/-- Exact ordered native block graphs for the complete source grid and frame.
FP32 primitives and explicit reduction plans preserve the original semantics. -/
theorem logsumexp_native_raw (ops : Ops) (plans : Nat → Plans)
    (a : RawKernel.Geometry) (scale : Word) (input : Nat → Word) (s : State Word)
    (binding : ∀ row,row<a.rows → ∀ col,col<a.columns →
      (s.mem "input" (row*a.columns+col)).read .real=input (row*a.columns+col))
    (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    ∃ t,RawKernel.run (fun j=>engine ops (plans j)) a scale jobs s=some t ∧
      (∀ row segment,row<a.rows → segment<a.blocks →
        t.mem "scratch" (row*a.blocks+segment)=.mk .real
          (blockValue (engine ops (plans (row*a.blocks+segment))) input
            scale row segment a.columns a.block a.hasScale)) ∧
      (∀ (r : RegionName) i,(r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,ran,values,frame⟩ := RawKernel.ordered (fun j=>engine ops (plans j)) a scale s jobs schedule
  refine ⟨t,ran,?_,frame⟩
  intro row segment hr hb
  have bound := RawKernel.output_offset_bound a row segment hr hb
  have positions := RawKernel.full_coverage a row segment hr hb
  rw [values _ bound]
  simp only [RawKernel.compute,positions.2.1,positions.2.2]
  rw [block_local _ _ input _ _ _ _ _ _ (binding row hr)]

#axiomsClean logsumexp_native_raw
#specNonCircular blockValue avoiding [rowBlock,Source.rowValue,RawKernel.run]
end
end VeriTile.Bench.Repairs.LogsumexpFwd.RawNative
