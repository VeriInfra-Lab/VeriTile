import bench.tritonbench_g.swiglu_fwd.Kernels
import bench.tritonbench_g.swiglu_fwd.Proofs.Mathematical_WholeTile

namespace VeriTile.Bench.Repairs.SwigluFwd.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- Every active output and every exterior typed memory cell. -/
def Result {α : Type} [Inhabited α] (f : α → α → α) (c : Layout)
    (B row tile : Nat) (entry final : State α) : Prop :=
  (∀ j : Fin B, tile * B + j.val < c.cols →
    final.mem c.OUT (row*c.so+(tile*B+j.val)) = Cell.mk .real
      (f ((entry.mem c.X (row*c.sx+(tile*B+j.val))).read .real)
         ((entry.mem c.Y (row*c.sy+(tile*B+j.val))).read .real))) ∧
  (∀ r o, Outside c B row tile r o → final.mem r o = entry.mem r o)

theorem run {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B row tile : Nat) (entry : State α) :
    ∃ final, exec M (kernel c.X c.Y c.OUT c.sx c.sy c.so c.cols B)
      (tileState M c B row tile entry) = some final ∧
      Result (scalar M) c B row tile entry final := by
  refine ⟨(sourceRun M c B row tile entry).val,
    (sourceRun M c B row tile entry).property, ?_, ?_⟩
  · intro j hj
    exact tile_readback M c B row tile entry ⟨j, PUnit.unit⟩ hj
  · exact tile_frame M c B row tile entry

#axiomsClean run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd.KernelProof
