import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealForwardValues
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def alpha (old : R) (m : ℝ) : ℝ :=
  match old with | none=>0 | some old=>Real.exp (old-m)
def weight (row start : Nat) (score : Nat → ℝ) (m : ℝ) (j : Nat) : ℝ :=
  if start+j ≤ row then Real.exp (score (start+j)-m) else 0

theorem maximum_finite (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : Fin p.blockM) (a : ForwardAccumulator R p)
    (hBN : 0<p.blockN) (hb : forwardHeadRow p s+c*p.blockN<p.rows)
    (hc : c*p.blockN ≤ forwardQueryRow p s i) :
    ∃ m,forwardMax model p s c a (i,())=some m := by
  let v:=logicalScore p s μ i (c*p.blockN)
  have hs : forwardScore model p s c (i,⟨0,hBN⟩,())=some v := by
    simpa [v,hc] using score_real p s μ h c (i,⟨0,hBN⟩,()) (by simpa using hb)
  have hle : (v : WithBot ℝ)≤forwardMax model p s c a (i,()) := by
    change (v : WithBot ℝ)≤max (Finset.univ.sup (fun j : Fin p.blockN=>forwardScore model p s c (i,j,()))) (a.maximum (i,()))
    rw [←show forwardScore model p s c (i,⟨0,hBN⟩,())=(v : WithBot ℝ) from hs]
    exact le_trans (Finset.le_sup (f:=fun j : Fin p.blockN=>forwardScore model p s c (i,j,()))
      (Finset.mem_univ (⟨0,hBN⟩ : Fin p.blockN))) (le_max_left _ _)
  cases he : forwardMax model p s c a (i,()) with
  | none=>rw [he] at hle; change (v : WithBot ℝ)≤⊥ at hle; simp at hle
  | some m=>exact ⟨m,rfl⟩

theorem weight_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : Fin p.blockM) (a : ForwardAccumulator R p)
    (m : ℝ) (hm : forwardMax model p s c a (i,())=some m) (j : Fin p.blockN)
    (hb : forwardHeadRow p s+c*p.blockN+j.val<p.rows) :
    forwardWeights model p s c a (i,j,())=some
      (weight (forwardQueryRow p s i) (c*p.blockN) (logicalScore p s μ i) m j.val) := by
  simp only [forwardWeights,hm]
  rw [score_real p s μ h c (i,j,()) hb]
  by_cases hv : c*p.blockN+j.val ≤ forwardQueryRow p s i <;>
    simp [model,algebra,exp,weight,hv]

theorem oldDenominator_real (p : ForwardParameters) (s : State R) (c : Nat)
    (i : Fin p.blockM) (a : ForwardAccumulator R p) (m L : ℝ)
    (hm : forwardMax model p s c a (i,())=some m) (hl : a.denominator (i,())=some L) :
    forwardOldDenominator model p s c a (i,())=some (L*alpha (a.maximum (i,())) m) := by
  simp only [forwardOldDenominator,hm,hl]
  cases he : a.maximum (i,()) <;> simp [model,algebra,exp,alpha,he,WithBot.realMul]

theorem denominator_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : Fin p.blockM) (a : ForwardAccumulator R p)
    (hb : ∀ j : Fin p.blockN, forwardHeadRow p s+c*p.blockN+j.val<p.rows)
    (m L : ℝ) (hm : forwardMax model p s c a (i,())=some m)
    (hl : a.denominator (i,())=some L) :
    forwardDenominator model p s c a (i,())=some
      (Mathematical.blockMass (forwardQueryRow p s i) (c*p.blockN) p.blockN (logicalScore p s μ i) m +
        L*alpha (a.maximum (i,())) m) := by
  change WithBot.realAdd (sum p.blockN (fun j=>forwardWeights model p s c a (i,j,())))
    (forwardOldDenominator model p s c a (i,()))=_
  have hw := funext (fun j=>weight_real p s μ h c i a m hm j (hb j))
  rw [hw,sum_some,oldDenominator_real p s c i a m L hm hl]
  rfl

theorem normalized_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : Fin p.blockM) (a : ForwardAccumulator R p)
    (m L : ℝ) (hm : forwardMax model p s c a (i,())=some m)
    (hl : forwardDenominator model p s c a (i,())=some L) (j : Fin p.blockN)
    (hb : forwardHeadRow p s+c*p.blockN+j.val<p.rows) :
    forwardNormalized model p s c a (i,j,())=some
      (weight (forwardQueryRow p s i) (c*p.blockN) (logicalScore p s μ i) m j.val/L) := by
  simp only [forwardNormalized,forwardReciprocal,hl]
  rw [weight_real p s μ h c i a m hm j hb]
  simp [model,algebra,WithBot.realMul,WithBot.realDiv,div_eq_mul_inv]

@[simp] theorem toFloat_ofFloat_real (d : FloatDType) (x : R) :
    toFloat d (ofFloat d x)=x := by cases d <;> rfl

theorem valueDot_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : FQuery p) (a : ForwardAccumulator R p)
    (hb : ∀ j : Fin p.blockN, forwardHeadRow p s+c*p.blockN+j.val<p.rows)
    (m L : ℝ) (hm : forwardMax model p s c a (i.1,())=some m)
    (hl : forwardDenominator model p s c a (i.1,())=some L) :
    forwardValueDot model p s c a i=some
      (Mathematical.blockNumerator (forwardQueryRow p s i.1) (c*p.blockN) p.blockN
        (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m/L) := by
  change sum p.blockN (fun j=>WithBot.realMul
    (toFloat p.dtype (ofFloat p.dtype (forwardNormalized model p s c a (i.1,j,()))))
    (toFloat p.dtype (forwardKV p s p.v c (j,i.2.1,()))))=_
  simp only [toFloat_ofFloat_real]
  have he : (fun j : Fin p.blockN=>WithBot.realMul (forwardNormalized model p s c a (i.1,j,()))
      (toFloat p.dtype (forwardKV p s p.v c (j,i.2.1,())))) =
      (fun j=>some ((if c*p.blockN+j.val ≤ forwardQueryRow p s i.1
        then Real.exp (logicalScore p s μ i.1 (c*p.blockN+j.val)-m)*vAt p s μ (c*p.blockN+j.val) i.2.1.val else 0)/L)) := by
    funext j
    rw [normalized_real p s μ h c i.1 a m L hm hl j (hb j),kv_real p s μ h p.v (by simp) c (j,i.2.1,()) (hb j)]
    by_cases hv : c*p.blockN+j.val ≤ forwardQueryRow p s i.1 <;>
      simp [weight,hv,WithBot.realMul,vAt,forwardKVOffset,Nat.add_assoc,div_mul_eq_mul_div]
  rw [he,sum_some,←Finset.sum_div]
  rfl

theorem numerator_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : FQuery p) (a : ForwardAccumulator R p)
    (hb : ∀ j : Fin p.blockN, forwardHeadRow p s+c*p.blockN+j.val<p.rows)
    (m oldL newL A : ℝ) (hm : forwardMax model p s c a (i.1,())=some m)
    (hl : a.denominator (i.1,())=some oldL) (hn : forwardDenominator model p s c a (i.1,())=some newL)
    (ha : a.numerator i=some A) :
    (forwardAdvance model p s c a).numerator i=some
      (A*(oldL*alpha (a.maximum (i.1,())) m/newL)+
        Mathematical.blockNumerator (forwardQueryRow p s i.1) (c*p.blockN) p.blockN
          (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m/newL) := by
  simp only [forwardAdvance,forwardRescaled,forwardReciprocal,ha,hn]
  rw [oldDenominator_real p s c i.1 a m oldL hm hl,valueDot_real p s μ h c i a hb m newL hm hn]
  simp [model,algebra,WithBot.realAdd,WithBot.realMul,WithBot.realDiv,div_eq_mul_inv]

#axiomsClean maximum_finite
#axiomsClean denominator_real
#axiomsClean numerator_real
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
