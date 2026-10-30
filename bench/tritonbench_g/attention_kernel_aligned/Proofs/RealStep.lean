import bench.tritonbench_g.attention_kernel_aligned.Proofs.RealBackend
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Real
open VeriTile Triton FP.Structural Source Full
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def mass (BN N base : Nat) (score : Nat → ℝ) (m : ℝ) (j : Fin BN) : ℝ :=
  if base+j.val<N then pow2 (score (base+j.val)-m) else 0

def alpha (old : R) (m : ℝ) : ℝ :=
  match old with | none=>0 | some old=>pow2 (old-m)

theorem maximum_finite (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : Fin p.blockM) (a : Accumulator R p) (hi : row p s i<p.length)
    (hBN : 0<p.blockN) (hc : c*p.blockN<p.length) :
    ∃ m,newMax model p s c a (i,())=some m := by
  let v:=logicalScore p s μ i (c*p.blockN)
  have hs : score32 model p s c (i,⟨0,hBN⟩,())=some v := by
    simpa [v,hc] using score_real p s μ h c (i,⟨0,hBN⟩,()) hi hBN
  have hle : (v : WithBot ℝ)≤newMax model p s c a (i,()) := by
    change (v : WithBot ℝ)≤max (a.maximum (i,())) (Finset.univ.sup (fun j : Fin p.blockN=>score32 model p s c (i,j,())))
    have hs' : score32 model p s c (i,⟨0,hBN⟩,())=(v : WithBot ℝ) := hs
    rw [←hs']
    exact le_trans (Finset.le_sup (f:=fun j : Fin p.blockN=>score32 model p s c (i,j,()))
      (Finset.mem_univ (⟨0,hBN⟩ : Fin p.blockN))) (le_max_right _ _)
  cases he : newMax model p s c a (i,()) with
  | none=>
    rw [he] at hle
    change (v : WithBot ℝ)≤⊥ at hle
    simp at hle
  | some m=>exact ⟨m,rfl⟩

theorem probability_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : Fin p.blockM) (a : Accumulator R p) (hi : row p s i<p.length)
    (hBN : 0<p.blockN) (m : ℝ) (hm : newMax model p s c a (i,())=some m) (j : Fin p.blockN) :
    weights model p s c a (i,j,())=some (mass p.blockN p.length (c*p.blockN) (logicalScore p s μ i) m j) := by
  simp only [weights,hm]
  rw [score_real p s μ h c (i,j,()) hi hBN]
  by_cases hj : c*p.blockN+j.val<p.length <;> simp [model,algebra,exp2,mass,hj]

theorem rescale_real (p : Parameters) (s : State R) (c : Nat) (i : Fin p.blockM) (a : Accumulator R p)
    (m : ℝ) (hm : newMax model p s c a (i,())=some m) :
    alphaValue model p s c a (i,())=some (alpha (a.maximum (i,())) m) := by
  simp only [alphaValue,hm]
  cases he : a.maximum (i,()) <;> simp [model,algebra,exp2,alpha,he]

theorem normalizer_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : Fin p.blockM) (a : Accumulator R p) (hi : row p s i<p.length)
    (hBN : 0<p.blockN) (m L : ℝ) (hm : newMax model p s c a (i,())=some m)
    (hl : a.denominator (i,())=some L) :
    (advance model p s c a).denominator (i,())=
      some (L*alpha (a.maximum (i,())) m+∑ j,mass p.blockN p.length (c*p.blockN) (logicalScore p s μ i) m j) := by
  change WithBot.realAdd (WithBot.realMul (a.denominator (i,())) (alphaValue model p s c a (i,())))
    (sum p.blockN (fun j=>weights model p s c a (i,j,())))=_
  rw [hl,rescale_real p s c i a m hm]
  have hp : (fun j : Fin p.blockN=>weights model p s c a (i,j,()))=
      (fun j=>some (mass p.blockN p.length (c*p.blockN) (logicalScore p s μ i) m j)) :=
    funext (probability_real p s μ h c i a hi hBN m hm)
  rw [hp,sum_some]
  rfl

theorem contribution_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : QLane p) (a : Accumulator R p) (hi : row p s i.1<p.length)
    (hBN : 0<p.blockN) (m : ℝ) (hm : newMax model p s c a (i.1,())=some m) :
    weightedDot model p s c a i=some
      (∑ j,mass p.blockN p.length (c*p.blockN) (logicalScore p s μ i.1) m j*vAt p s μ (c*p.blockN+j.val) i.2.1.val) := by
  change sum p.blockN (fun j=>WithBot.realMul (weights model p s c a (i.1,j,()))
    (vValue model p s c (j,i.2.1,())))=_
  have hp : (fun j : Fin p.blockN=>WithBot.realMul (weights model p s c a (i.1,j,()))
      (vValue model p s c (j,i.2.1,())))=
      (fun j=>some (mass p.blockN p.length (c*p.blockN) (logicalScore p s μ i.1) m j*vAt p s μ (c*p.blockN+j.val) i.2.1.val)) := by
    funext j
    rw [probability_real p s μ h c i.1 a hi hBN m hm j,value_real p s μ h c (j,i.2.1,())]
    by_cases hj : c*p.blockN+j.val<p.length <;> simp [mass,hj,WithBot.realMul]
  rw [hp,sum_some]

theorem accumulator_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : QLane p) (a : Accumulator R p) (hi : row p s i.1<p.length)
    (hBN : 0<p.blockN) (m A : ℝ)
    (hm : newMax model p s c a (i.1,())=some m) (ha : a.numerator i=some A) :
    (advance model p s c a).numerator i=some
      (A*alpha (a.maximum (i.1,())) m+
        ∑ j,mass p.blockN p.length (c*p.blockN) (logicalScore p s μ i.1) m j*vAt p s μ (c*p.blockN+j.val) i.2.1.val) := by
  change WithBot.realAdd (WithBot.realMul (a.numerator i) (alphaValue model p s c a (i.1,())))
    (weightedDot model p s c a i)=_
  rw [ha,rescale_real p s c i.1 a m hm,contribution_real p s μ h c i a hi hBN m hm]
  rfl
#axiomsClean maximum_finite
#axiomsClean normalizer_real
#axiomsClean accumulator_real
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Real
