import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Frames

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

def correctedValue {α : Type} (A : TensorArithmetic α) (scale : ℝ) (loss lse : α) : α :=
  if scale≠0 then A.add loss (A.scale scale (A.square lse)) else loss

theorem correction_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt)
    (scale : ℝ) (rows : Nat) (losses lse output : Tensor) (initial final : World α σ)
    (lv sv : Nat → α) (loss : FloatArray obs initial losses [rows] lv)
    (saved : FloatArray obs initial lse [rows] sv) (separate : lse.region≠losses.region)
    (trace : CorrectionTrace rt scale losses lse initial output final) :
    output=losses ∧ Mutation obs initial final losses ∧
      FloatArray obs final output [rows] (fun row => correctedValue A scale (lv row) (sv row)) ∧
      FloatArray obs final lse [rows] sv := by
  cases trace with
  | zero zero =>
    refine ⟨rfl,Mutation.refl obs initial losses,?_,saved⟩
    simpa only [correctedValue,zero,ne_eq,not_true_eq_false,if_false] using loss
  | nonzero nonzero squared scaled result ss sm closed squareCall scaleCall addCall =>
    have sq := ops.square lse initial squared ss saved.live saved.dtype squareCall
    have mul := ops.scalarMul scale squared ss scaled sm sq.allocation.live sq.allocation.dtypeEq scaleCall
    have prepFrame := sq.allocation.frame.trans mul.allocation.frame
    have preserved := loss.preserve prepFrame
    have sameShape : losses.shape=scaled.shape :=
      loss.shapeEq.trans (mul.allocation.shapeEq.trans (sq.allocation.shapeEq.trans saved.shapeEq)).symm
    have addition := ops.add losses scaled sm _ _ preserved.live mul.allocation.live
      preserved.dtype mul.allocation.dtypeEq sameShape addCall
    obtain ⟨identity,mutation,values⟩ := addition
    refine ⟨identity,mutation.prepend prepFrame,?_,saved.mutate (mutation.prepend prepFrame) separate⟩
    rw [identity]
    refine ⟨loss.shapeEq,loss.dtype,mutation.live _ preserved.live,?_⟩
    intro row hr
    have scaledBound : row<scaled.numel := by simpa only [Tensor.numel,← sameShape] using hr
    have squareShape : squared.shape=losses.shape := sq.allocation.shapeEq.trans (saved.shapeEq.trans loss.shapeEq.symm)
    have squareBound : row<squared.numel := by simpa only [Tensor.numel,squareShape] using hr
    have savedBound : row<lse.numel := by simpa only [Tensor.numel,saved.shapeEq,loss.shapeEq] using hr
    rw [values row hr,preserved.values row hr,mul.values row scaledBound,sq.values row squareBound,
      saved.values row savedBound]
    simp only [correctedValue,if_pos nonzero]

#axiomsClean correction_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
