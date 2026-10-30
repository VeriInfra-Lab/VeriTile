import bench.tritonbench_g.softmax_flaggems.Proofs.IO
import bench.tritonbench_g.softmax_flaggems.Proofs.RealResults

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option maxHeartbeats 3000000

theorem tensorIndex_address (d : Dimensions) (row : Fin d.M) (n : Fin d.N) (col : Fin d.K) :
    (tensorIndex d ((row,n),col)).val = Target.address d.N d.K row.val col.val n.val := by
  change col.val+d.K*(n.val+d.N*row.val)=row.val*d.N*d.K+n.val*d.K+col.val
  ring

theorem forward_tensor_read (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (raw : (forwardIO X Y d c).Inputs) (row : Fin d.M) (n : Fin d.N) (col : Fin d.K) :
    extend (forwardInput X Y d c raw) (Target.address d.N d.K row.val col.val n.val) =
      tensorValues (WithBot ℝ) d X raw row n col := by
  rw [←tensorIndex_address]
  simp only [extend_at,forwardInput,tensorValues,Equiv.coe_fn_mk]

theorem backward_saved_read (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (raw : (backwardIO P G Y d c).Inputs) (row : Fin d.M) (n : Fin d.N) (col : Fin d.K) :
    extend (backwardInput P G Y d c raw 0) (Target.address d.N d.K row.val col.val n.val) =
      (backwardValues P G Y d c raw).saved row n col := by
  rw [←tensorIndex_address]
  simp only [extend_at,backwardInput,backwardValues,Equiv.coe_fn_mk,Fin.val_zero,↓reduceIte]

theorem backward_gradient_read (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (raw : (backwardIO P G Y d c).Inputs) (row : Fin d.M) (n : Fin d.N) (col : Fin d.K) :
    extend (backwardInput P G Y d c raw 1) (Target.address d.N d.K row.val col.val n.val) =
      (backwardValues P G Y d c raw).gradient row n col := by
  rw [←tensorIndex_address]
  simp only [extend_at,backwardInput,backwardValues,Equiv.coe_fn_mk,Fin.val_one,one_ne_zero,↓reduceIte]

/-- Whole-grid forward judgment equivalence, including the original finite
column domain, all input values, source safety, successful schedule and frame. -/
theorem logical_forward_iff (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hK : 0<d.K) :
    (logicalForwardIO X Y d c ⊨{LogicalForwardDomain X Y d c} fun xs row n col =>
      ((xs row n col).elim 0 Real.exp /
        ∑ j : Fin d.N, (xs row j col).elim 0 Real.exp : ℝ)) ↔
    (forwardIO X Y d c).ImplementsOn (ForwardDomain X Y d c) (forwardFormula X Y d c) := by
  have target_eq : (logicalForwardIO X Y d c).lift (fun xs row n col =>
      (((xs row n col).elim 0 Real.exp /
        ∑ j : Fin d.N, (xs row j col).elim 0 Real.exp : ℝ) : WithBot ℝ)) =
      forwardFormula X Y d c := by
    funext raw
    apply (tensorValues (WithBot ℝ) d Y).injective
    change tensorValues (WithBot ℝ) d Y ((tensorValues (WithBot ℝ) d Y).symm _) = _
    rw [Equiv.apply_symm_apply]
    funext row n col
    change _ = (Target.flatForward (extend (forwardInput X Y d c raw))
      d.N d.K (tensorIndex d ((row,n),col)).val : WithBot ℝ)
    rw [tensorIndex_address,Target.flatForward_coordinate _ _ _ _ _ _ hK col.isLt]
    simp only [Target.forward,Target.normalized,Target.denominator,Target.column,Target.mass,
      forward_tensor_read]
    rfl
  change (forwardIO X Y d c).ImplementsOn
    (fun raw => ForwardDomain X Y d c ((tensorValues (WithBot ℝ) d X).symm
      (tensorValues (WithBot ℝ) d X raw))) _ ↔ _
  simp only [Equiv.symm_apply_apply]
  rw [target_eq]

/-- The complete saved-output VJP judgment on logical M × N × K tensors. -/
theorem logical_backward_iff (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) :
    (logicalBackwardIO P G Y d c ⊨ fun xs row n col =>
      xs.saved row n col * (xs.gradient row n col - ∑ j : Fin d.N,
        xs.saved row j col * xs.gradient row j col)) ↔
    (backwardIO P G Y d c).Implements (backwardFormula P G Y d c) := by
  have target_eq : (logicalBackwardIO P G Y d c).lift (fun xs row n col =>
      xs.saved row n col * (xs.gradient row n col - ∑ j : Fin d.N,
        xs.saved row j col * xs.gradient row j col)) = backwardFormula P G Y d c := by
    funext raw
    apply (tensorValues ℝ d Y).injective
    change tensorValues ℝ d Y ((tensorValues ℝ d Y).symm _) = _
    rw [Equiv.apply_symm_apply]
    funext row n col
    change _ = Target.flatBackward (extend (backwardInput P G Y d c raw 0))
      (extend (backwardInput P G Y d c raw 1)) d.N d.K (tensorIndex d ((row,n),col)).val
    rw [tensorIndex_address]
    obtain ⟨hr,hc⟩ := Target.coordinates d.N d.K row.val col.val n.val hN hK n.isLt col.isLt
    simp only [Target.flatBackward,hr,hc,backward_saved_read,backward_gradient_read]
    rfl
  change (backwardIO P G Y d c).Implements _ ↔ _
  rw [target_eq]

#axiomsClean logical_forward_iff
#axiomsClean logical_backward_iff
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
