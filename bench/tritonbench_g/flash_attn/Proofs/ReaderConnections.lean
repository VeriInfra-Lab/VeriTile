import bench.tritonbench_g.flash_attn.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.ReaderConnections
open VeriTile Triton FP.Structural FlashAttn
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem whole_real (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (scale : ℝ)
    (score weight : LogicalIO.Inputs KernelReal.R x → Nat → Nat → Nat → Nat → ℝ)
    (F : LogicalIO.Inputs KernelReal.R x → LogicalIO.Outputs KernelReal.R x)
    (scoreDefinition : score = fun xs batch head row key =>
      ((List.range x.shape.dim).map (fun d =>
    (scale*1.44269504*LogicalIO.realAt xs.query batch head row d)*LogicalIO.realAt xs.key batch head key d)).sum)
    (weightDefinition : weight = fun xs batch head row key =>
      Real.exp (Real.log 2 * score xs batch head row key))
    (definition : F = fun xs =>
      let keys := fun row => (List.range x.shape.length).filter (visible x.shape row)
      let mass := fun batch head row =>
    ((keys row).map (fun key => weight xs batch head row key)).sum
  ⟨(fun batch head row col => some (
      ((keys row.val).map (fun key => weight xs batch.val head.val row.val key *
        LogicalIO.realAt xs.value batch.val head.val key col.val)).sum / mass batch.val head.val row.val)),
    fun batch head row => some (Real.log (mass batch.val head.val row.val)/Real.log 2)⟩) :
    LogicalIO.view KernelReal.algebra KernelReal.dot x (some scale) ⊨{LogicalIO.RealInputs x} F := by
  apply (LogicalIO.judgment_iff x scale _).mpr
  have formula : (fun raw => Fin.cases
      (fun i => (F (LogicalIO.inputs KernelReal.R x raw)).values i.1 i.2.1 i.2.2.1 i.2.2.2.1)
      (Fin.cases (fun i => (F (LogicalIO.inputs KernelReal.R x raw)).logMass i.1 i.2.1 i.2.2.1)
        (fun i => i.elim0))) = Logical.mathematical x scale := by
    funext raw port i
    fin_cases port <;>
      simp only [definition,weightDefinition,scoreDefinition,LogicalIO.query_at,LogicalIO.key_at,LogicalIO.value_at,
        Logical.mathematical,Logical.numerator,Logical.denominator,Logical.score,Spec.keys,pow2]
    all_goals rfl
  exact formula ▸ PublicIO.real_implements x hs ho scale

end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.ReaderConnections
