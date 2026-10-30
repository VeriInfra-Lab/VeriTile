import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.ReaderConnections
open VeriTile Triton
open scoped BigOperators VeriTile.Triton.LogicalIO VeriTile.Triton.LogicalDomainIO
noncomputable section

theorem forward (h : Call) (inv : (Fin h.cols → ℝ) → ℝ) (weight : ℝ → ℝ)
    (invDefinition : inv = fun x =>
      1/Real.sqrt ((∑ j, x j^2)/(h.cols : ℝ)+h.eps))
    (weightDefinition : weight = fun w =>
      if h.gemma then w+1 else w)
    (F : LogicalIO.Inputs h → LogicalIO.ForwardOutputs h)
    (definition : F = fun xs =>
      ⟨(fun row col => xs.x row col*inv (xs.x row)*weight (xs.weight col)),
      fun row => inv (xs.x row)⟩) :
    LogicalIO.forwardView h ⊨ F := by
  have formula : (LogicalIO.forwardView h).lift (F) = Reference.real h Bool.false := by
    funext raw port j
    fin_cases port
    · change (F (LogicalIO.inputs h raw)).values j.divNat j.modNat = _
      simp only [definition,LogicalIO.x_row,LogicalIO.weight_eq,invDefinition,weightDefinition,
        Reference.real,RealReference.output,RealReference.inv,RealReference.variance]
      rfl
    · change (F (LogicalIO.inputs h raw)).savedR j = _
      simp only [definition,LogicalIO.x_row,invDefinition,Reference.real,RealReference.inv,RealReference.variance]
      rfl
  exact Eq.mpr (congrArg (RealIO.program h Bool.false).Implements formula) (RealPublicIO.forward h)

theorem backward (h : Call) (inv : (Fin h.cols → ℝ) → ℝ) (weight : ℝ → ℝ)
    (invDefinition : inv = fun x =>
      1/Real.sqrt ((∑ j, x j^2)/(h.cols : ℝ)+h.eps))
    (weightDefinition : weight = fun w =>
      if h.gemma then w+1 else w)
    (F : LogicalIO.Inputs h → LogicalIO.BackwardOutputs h)
    (definition : F = fun xs =>
      ⟨(fun row col =>
      let x := xs.x row
      let g := xs.upstream row
      let r := inv x
      r*(g col*weight (xs.weight col))-(r^3/(h.cols : ℝ))*x col*
      (∑ k, g k*weight (xs.weight k)*x k)),
      fun col => ∑ row, xs.upstream row col*xs.x row col*inv (xs.x row)⟩)
    (domain : LogicalIO.Inputs h → Prop)
    (domainDefinition : domain = fun xs =>
      (∀ row, xs.savedR row=inv (xs.x row)) ∧
      (h.rows=0 → ∀ col, xs.initialDW col=0)) :
    LogicalIO.backwardView h ⊨{domain} F := by
  have domain : (fun raw => domain (LogicalIO.inputs h raw)) = Reference.realDomain h Bool.true := by
    funext raw
    simp only [domainDefinition,LogicalIO.x_row,LogicalIO.saved_eq,LogicalIO.initial_eq,
      Reference.realDomain,invDefinition,RealReference.inv,RealReference.variance,true_implies]
  have formula : (LogicalIO.backwardView h).lift (F) = Reference.real h Bool.true := by
    funext raw port j
    fin_cases port
    · change (F (LogicalIO.inputs h raw)).inputGradient j.divNat j.modNat = _
      simp only [definition,LogicalIO.x_row,LogicalIO.upstream_row,LogicalIO.weight_eq,invDefinition,weightDefinition,
        Reference.real,RealReference.inputGradient,RealReference.inv,RealReference.variance]
      rfl
    · change (F (LogicalIO.inputs h raw)).weightGradient j = _
      simp only [definition,LogicalIO.x_row,LogicalIO.upstream_row,invDefinition,Reference.real,
        RealReference.weightGradient,RealReference.inv,RealReference.variance]
      rfl
  exact Eq.mpr (congrArg₂ (RealIO.program h Bool.true).ImplementsOn domain formula) (RealPublicIO.backward h)

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.ReaderConnections
