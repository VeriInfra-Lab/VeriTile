import bench.tritonbench_g.triton_attention.Proofs.RealForwardCTA
import bench.tritonbench_g.triton_attention.Proofs.FPPreprocess
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def PreReads (p : PreParameters) (s : State R) (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) : Prop :=
  (∀ i : PreLane p,toFloat p.dtype ((s.mem p.output (preOffset p s i)).read p.dtype.toTileDType)=
    some (out (preRow p s (i.1,())) i.2.1.val)) ∧
  (∀ i : PreLane p,toFloat p.dtype ((s.mem p.gradient (preOffset p s i)).read p.dtype.toTileDType)=
    some (grad (preRow p s (i.1,())) i.2.1.val)) ∧
  (∀ i : PreRow p,(s.mem p.denominator (preRow p s i)).read .fp32=some (den (preRow p s i)))

theorem preQuotient_real (p : PreParameters) (s : State R)
    (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) (h : PreReads p s out grad den) (i : PreLane p) :
    preQuotient model p s i=some (grad (preRow p s (i.1,())) i.2.1.val/den (preRow p s (i.1,()))) := by
  change WithBot.realDiv (toFloat p.dtype ((s.mem p.gradient (preOffset p s i)).read p.dtype.toTileDType))
    ((s.mem p.denominator (preRow p s (i.1,()))).read .fp32)=_
  rw [h.2.1 i,h.2.2 (i.1,())]
  rfl

theorem preDelta_real (p : PreParameters) (s : State R)
    (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) (h : PreReads p s out grad den) (i : PreRow p) :
    preDelta model p s i=some ((∑ d ∈ Finset.range p.head,out (preRow p s i) d*grad (preRow p s i) d)/den (preRow p s i)) := by
  change sum p.head (fun d=>WithBot.realMul
    (toFloat p.dtype ((s.mem p.output (preOffset p s (i.1,d,()))).read p.dtype.toTileDType))
    (preQuotient model p s (i.1,d,())))=_
  have he : (fun d : Fin p.head=>WithBot.realMul
      (toFloat p.dtype ((s.mem p.output (preOffset p s (i.1,d,()))).read p.dtype.toTileDType))
      (preQuotient model p s (i.1,d,())))=
      (fun d=>some (out (preRow p s i) d.val*grad (preRow p s i) d.val/den (preRow p s i))) := by
    funext d
    rw [h.1 (i.1,d,()),preQuotient_real p s out grad den h (i.1,d,())]
    simp [WithBot.realMul,mul_div_assoc]
  rw [he,sum_some,←Finset.sum_div]
  congr 2
  exact Fin.sum_univ_eq_sum_range (fun d : Nat=>out (preRow p s i) d*grad (preRow p s i) d) p.head

#axiomsClean preQuotient_real
#axiomsClean preDelta_real
#specNonCircular PreReads avoiding [FP.Structural.exec,preDelta,preWrite]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
