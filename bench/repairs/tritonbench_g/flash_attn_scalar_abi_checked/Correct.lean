import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.ReaderConnections
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
open VeriTile Triton FP.Structural FlashAttn
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Q/K/V and O/L channels with independent strides and the complete source
grid. Runtime scale is bound to the original scalar register. -/
def attentionIO (x : Layout) (scale : ℝ) :=
  LogicalIO.view KernelReal.algebra KernelReal.dot x (some scale)

/-- Exact decimal score conversion retained from the source, over the
logical head dimension. Storage/FP16 tags are erased only in this Real layer. -/
def attentionScore (x : Layout) (scale : ℝ) (xs : LogicalIO.Inputs KernelReal.R x)
    (batch head row key : Nat) : ℝ :=
  ((List.range x.shape.dim).map (fun d =>
    (scale*1.44269504*LogicalIO.realAt xs.query batch head row d)*LogicalIO.realAt xs.key batch head key d)).sum

/-- Every visible logical key participates; causal mode keeps key≤row. -/
def attentionWeight (x : Layout) (scale : ℝ) (xs : LogicalIO.Inputs KernelReal.R x)
    (batch head row key : Nat) : ℝ :=
  Real.exp (Real.log 2 * attentionScore x scale xs batch head row key)

/-- Two complete output tensors: normalized weighted V, and base-2 log mass.
Both are ordinary reals (`some`); an undefined observation is never collapsed to zero. -/
def attentionFormula (x : Layout) (scale : ℝ) (xs : LogicalIO.Inputs KernelReal.R x) :
    LogicalIO.Outputs KernelReal.R x :=
  let keys := fun row => (List.range x.shape.length).filter (visible x.shape row)
  let mass := fun batch head row =>
    ((keys row).map (fun key => attentionWeight x scale xs batch head row key)).sum
  ⟨(fun batch head row col => some (
      ((keys row.val).map (fun key => attentionWeight x scale xs batch.val head.val row.val key *
        LogicalIO.realAt xs.value batch.val head.val key col.val)).sum / mass batch.val head.val row.val)),
    fun batch head row => some (Real.log (mass batch.val head.val row.val)/Real.log 2)⟩

/-- The complete attention tensor and log2 denominator implement independent
Real formulas. Ordinary Real inputs and the original geometry are explicit;
source access safety, successful execution and the exact exterior frame are
conclusions of the shared IO judgment. -/
specification whole_real (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (scale : ℝ) :
    Spec.Real ((attentionIO x scale) ⊨{LogicalIO.RealInputs x} (attentionFormula x scale)) :=
  ReaderConnections.whole_real x hs ho scale (attentionScore x scale) (attentionWeight x scale)
    (attentionFormula x scale) rfl rfl rfl

#axiomsClean whole_real
#stmtSurfaceSubset whole_real ⊆
  [LogicalIO.RealInputs,LogicalIO.Inputs,LogicalIO.Outputs,Spec.Real,LogicalProgramIO.ImplementsOn,attentionIO,Logical.RealInputs,Logical.mathematical,attentionFormula,
   Layout,Layout.shape,Shape.Valid,OutputInjective,KernelReal.R]
#specNonCircular attentionFormula avoiding
  [FP.Structural.exec,FP.Structural.run,ActualSource.runGrid,ActualSource.referenceGrid,ActualSource.accumulated]
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
