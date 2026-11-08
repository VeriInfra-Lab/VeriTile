import bench.tritonbench_g.attention_fwd_triton1.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Correct
open VeriTile Triton FP.Structural
open AttentionFwdTriton1.Whole001
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option autoImplicit false

/-- O is chunk-causal (all keys in the current 32-row chunk); H is the
pre-update KᵀV prefix. STORE masks the H channel in the shared complete IO. -/
def formula (c : Shape) (xs : LogicalIO.Inputs c) : LogicalIO.Outputs c where
  output := fun head row col =>
    (1 / Real.sqrt (c.D : ℝ)) *
      ∑ u ∈ Finset.range (min ((row.val / 32 + 1) * 32) c.T),
        ∑ e ∈ Finset.range c.D,
          xs.query head.val row.val e * xs.key head.val u e * xs.value head.val u col.val
  history := fun head chunk row col =>
    ∑ u ∈ Finset.range (min (chunk.val * 32) c.T),
      xs.key head.val u row.val * xs.value head.val u col.val

/-- Independent chunked linear-attention output and pre-update history,
with actual source access safety, successful all-head/all-chunk execution and
complete frame. Both STORE and IFCOND branches retain their exact IO masks. -/
specification kernel_real (c : Shape) (storeFlag ifcond : Bool) :
    Spec.Real (LogicalIO.io c storeFlag ifcond ⊨ formula c) :=
  LogicalReader.correct c storeFlag ifcond _ (by intros; rfl) (by intros; rfl)

#stmtSurfaceSubset kernel_real ⊆ [Spec.Real,LogicalIO.io,LogicalIO.Inputs,LogicalIO.Outputs,formula,LogicalProgramIO.Implements,Shape,Term,instInhabitedTerm]
#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,SharedGrid.result,Reference.output,Reference.carry]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Correct
