import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.AggregateTrace
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Mask

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Observations TensorArithmetic TensorLibrary FloatArray IntArray
  Preserves Mutation value fresh_ne_live)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

def correctionZ {α : Type} (A : TensorArithmetic α) (o : Options) (target : Int) (lse : α) : α :=
  if o.squareScale≠0 then (if target=o.ignored then A.zero else A.scale o.squareScale (A.square lse)) else A.zero

def correctionLoss {α : Type} (A : TensorArithmetic α) (o : Options) (target : Int) (loss lse : α) : α :=
  if o.squareScale≠0 then A.add loss (correctionZ A o target lse) else loss

/-- The logging result is a separate actual tensor. Its masked scratch writes
preserve all pre-existing allocations; the final addition mutates only loss. -/
theorem correction_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt.toRuntime)
    (zeros : ZeroAPI obs rt A.zero) (o : Options) (rows : Nat) (labels losses lse : Tensor)
    (initial final : World α σ) (output : Tensor × Tensor) (lv sv : Nat → α) (targets : Nat → Int)
    (loss : FloatArray obs initial losses [rows] lv) (saved : FloatArray obs initial lse [rows] sv)
    (label : IntArray obs initial labels [rows] targets) (separate : lse.region≠losses.region)
    (trace : CorrectionTrace rt o labels losses lse initial output final) :
    output.1=losses ∧ Mutation obs initial final losses ∧
      FloatArray obs final output.1 [rows] (fun row => correctionLoss A o (targets row) (lv row) (sv row)) ∧
      FloatArray obs final output.2 [rows] (fun row => correctionZ A o (targets row) (sv row)) ∧
      FloatArray obs final lse [rows] sv ∧ output.2.region≠losses.region := by
  cases trace with
  | zero zero z after call =>
    have produced := zeros.zeros losses initial z final loss.live loss.dtype call
    have zValues : FloatArray obs final z [rows] (fun _ => A.zero) := by
      simpa only [loss.shapeEq] using produced.array
    have zSeparate := fresh_ne_live obs _ _ _ _ _ _ produced.allocation _ loss.live
    refine ⟨rfl,produced.allocation.frame.mutation losses,?_,?_,saved.preserve produced.allocation.frame,zSeparate⟩
    · simpa only [correctionLoss,zero,ne_eq,not_true_eq_false,↓reduceIte] using loss.preserve produced.allocation.frame
    · simpa only [correctionZ,zero,ne_eq,not_true_eq_false,↓reduceIte] using zValues
  | nonzero nonzero squared scaled mask z result ss sm compared masked after squareCall scaleCall compareCall maskCall addCall =>
    have sq := ops.square lse initial squared ss saved.live saved.dtype squareCall
    have mul := ops.scalarMul o.squareScale squared ss scaled sm sq.allocation.live sq.allocation.dtypeEq scaleCall
    have prep := sq.allocation.frame.trans mul.allocation.frame
    have scaledShape : scaled.shape=[rows] := mul.allocation.shapeEq.trans (sq.allocation.shapeEq.trans saved.shapeEq)
    have scaledValues : FloatArray obs sm scaled [rows] (fun row => A.scale o.squareScale (A.square (sv row))) := by
      refine ⟨scaledShape,mul.allocation.dtypeEq,mul.allocation.live,?_⟩
      intro row hr
      have squareBound : row<squared.numel := by simpa only [Tensor.numel,mul.allocation.shapeEq] using hr
      have savedBound : row<lse.numel := by simpa only [Tensor.numel,scaledShape,saved.shapeEq] using hr
      rw [mul.values row hr,sq.values row squareBound,saved.values row savedBound]
    have masking := CrossEntropy1.Host.mask_values obs A rt.toRuntime ops rows o.ignored scaled labels mask z
      sm compared masked _ targets scaledValues (label.preserve prep) compareCall maskCall
    have fresh : ¬initial.live scaled.region := fun live => mul.allocation.fresh (sq.allocation.frame.live _ live)
    have maskedFrame := prep.freshMutation masking.2.1 fresh
    have preserved := loss.preserve maskedFrame
    have addition := ops.add losses z masked result final preserved.live masking.2.2.live
      preserved.dtype masking.2.2.dtype (loss.shapeEq.trans masking.2.2.shapeEq.symm) addCall
    have separateZ : z.region≠losses.region := by
      rw [masking.1]
      exact fresh_ne_live obs _ _ _ _ _ _ mul.allocation _ (sq.allocation.frame.live _ loss.live)
    have mutation := addition.2.1.prepend maskedFrame
    refine ⟨addition.1,mutation,?_,?_,saved.mutate mutation separate,separateZ⟩
    · rw [addition.1]
      refine ⟨loss.shapeEq,loss.dtype,addition.2.1.live _ preserved.live,?_⟩
      intro row hr
      have zBound : row<z.numel := by simpa only [Tensor.numel,loss.shapeEq,masking.2.2.shapeEq] using hr
      rw [addition.2.2 row hr,preserved.values row hr,masking.2.2.values row zBound]
      simp only [correctionLoss,correctionZ,if_pos nonzero]
    · simpa only [correctionZ,if_pos nonzero] using masking.2.2.mutate addition.2.1 separateZ

#axiomsClean correction_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
