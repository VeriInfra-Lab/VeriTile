import bench.tritonbench_g.bgmv_expand_slice.Proofs.IOReal
import bench.tritonbench_g.bgmv_expand_slice.IO

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem logical_iff (h : Call) :
    (gemvLogicalIO h ⊨{logicalValid} fun xs batch row =>
      (∑ k : Fin h.K, xs.inputAt batch k * xs.weight batch row k) +
        if h.add then xs.previous batch row else 0) ↔
    (sourceIO Real.algebra Real.reduction id h).ImplementsOn Valid (PublicMath.formula h) := by
  change (sourceIO Real.algebra Real.reduction id h).ImplementsOn
    (fun vals => logicalValid (logicalInputsEquiv h vals))
    (fun vals => (logicalOutputsEquiv h).symm (fun batch row =>
      (∑ k : Fin h.K, (logicalInputsEquiv h vals).inputAt batch k *
        (logicalInputsEquiv h vals).weight batch row k) +
        if h.add then (logicalInputsEquiv h vals).previous batch row else 0)) ↔ _
  have domain_eq : (fun vals => logicalValid (logicalInputsEquiv h vals)) = (Valid : Inputs ℝ h → Prop) := rfl
  change (sourceIO Real.algebra Real.reduction id h).ImplementsOn Valid _ ↔ _
  have target_eq : (fun vals => (logicalOutputsEquiv h).symm (fun batch row =>
      (∑ k : Fin h.K, (logicalInputsEquiv h vals).inputAt batch k *
        (logicalInputsEquiv h vals).weight batch row k) +
        if h.add then (logicalInputsEquiv h vals).previous batch row else 0)) = PublicMath.formula h := by
    funext vals o
    fin_cases o
    funext j
    rcases j with ⟨b,row,⟨⟩⟩
    by_cases add : h.add = Bool.true <;> simp [add,logicalOutputsEquiv,Equiv.symm,
      logicalInputsEquiv,LogicalInputs.inputAt,PublicMath.formula,result,input,weight,previous,
      show (default : ℝ)=0 from rfl] <;> rfl
  exact (congrArg (fun f => (sourceIO Real.algebra Real.reduction id h).ImplementsOn Valid f) target_eq).to_iff

theorem logical_implements (h : Call) :
    gemvLogicalIO h ⊨{logicalValid} fun xs batch row =>
      (∑ k : Fin h.K, xs.inputAt batch k * xs.weight batch row k) +
        if h.add then xs.previous batch row else 0 :=
  (logical_iff h).mpr (PublicMath.implements h)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
