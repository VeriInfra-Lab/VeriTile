import bench.tritonbench_g.attention_forward_triton.Proofs.RealBackend
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Real
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def logicalScore (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (i : Fin 128) (j : Nat) : ℝ :=
  RealMath.score (qAt p s μ) (kAt p s μ) (qsAt p s μ) (ksAt p s μ) (s.pids 0*128+i.val) j

def mass (N row : Nat) (score : Nat → ℝ) (m : ℝ) (j : Fin 64) : ℝ :=
  if j.val<N-row then pow2 (score (row+j.val)-m) else 0

def alpha (old : R) (m : ℝ) : ℝ :=
  match old with | none=>0 | some old=>pow2 (old-m)

theorem maximum_finite (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : Fin 128) (a : Acc R) (hi : s.pids 0*128+i.val<p.length)
    (halign : row%64=0) (hrow : row<p.length) :
    ∃ m,Source.maximum algebra dotModel p s row a i=some m := by
  let v:=logicalScore p s μ i row
  have hj : (0:Nat)<p.length-row := by omega
  have hs : scores algebra dotModel p s row (i,⟨0,by decide⟩,())=some v := by
    simpa [logicalScore,v,hj] using score_real p s μ h row (i,⟨0,by decide⟩,()) hi halign
  have hle : (v : WithBot ℝ)≤Source.maximum algebra dotModel p s row a i := by
    change (v : WithBot ℝ)≤max (a.maximum i) (Finset.univ.sup (fun j : Fin 64=>scores algebra dotModel p s row (i,j,())))
    have hs' : scores algebra dotModel p s row (i,⟨0,by decide⟩,())=(v : WithBot ℝ) := hs
    rw [←hs']
    exact le_trans (Finset.le_sup (f:=fun j : Fin 64=>scores algebra dotModel p s row (i,j,()))
      (Finset.mem_univ (⟨0,by decide⟩ : Fin 64))) (le_max_right _ _)
  cases he : Source.maximum algebra dotModel p s row a i with
  | none=>
    rw [he] at hle
    change (v : WithBot ℝ)≤⊥ at hle
    simp at hle
  | some m=>exact ⟨m,rfl⟩

theorem probability_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : Fin 128) (a : Acc R) (hi : s.pids 0*128+i.val<p.length)
    (halign : row%64=0) (m : ℝ) (hm : Source.maximum algebra dotModel p s row a i=some m) (j : Fin 64) :
    probability algebra dotModel p s row a (i,j,())=some (mass p.length row (logicalScore p s μ i) m j) := by
  simp only [probability,hm]
  rw [score_real p s μ h row (i,j,()) hi halign]
  by_cases hj : j.val<p.length-row <;> simp [algebra,exp2,mass,logicalScore,hj]

theorem rescale_real (p : Parameters) (s : State R) (row : Nat) (i : Fin 128) (a : Acc R)
    (m : ℝ) (hm : Source.maximum algebra dotModel p s row a i=some m) :
    rescale algebra dotModel p s row a i=some (alpha (a.maximum i) m) := by
  simp only [rescale,hm]
  cases he : a.maximum i <;> simp [algebra,exp2,alpha,he]

theorem normalizer_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : Fin 128) (a : Acc R) (hi : s.pids 0*128+i.val<p.length)
    (halign : row%64=0) (m L : ℝ) (hm : Source.maximum algebra dotModel p s row a i=some m)
    (hl : a.normalizer i=some L) :
    (update algebra dotModel p s row a).normalizer i=
      some (L*alpha (a.maximum i) m+∑ j,mass p.length row (logicalScore p s μ i) m j) := by
  change WithBot.realAdd (WithBot.realMul (a.normalizer i) (rescale algebra dotModel p s row a i))
    (sum 64 (fun j=>probability algebra dotModel p s row a (i,j,())))=_
  rw [hl,rescale_real p s row i a m hm]
  have hp : (fun j : Fin 64=>probability algebra dotModel p s row a (i,j,()))=
      (fun j=>some (mass p.length row (logicalScore p s μ i) m j)) :=
    funext (probability_real p s μ h row i a hi halign m hm)
  rw [hp,sum_some]
  rfl

theorem contribution_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [128,128]) (a : Acc R) (hi : s.pids 0*128+i.1.val<p.length)
    (hd : i.2.1.val<96) (halign : row%64=0) (m : ℝ)
    (hm : Source.maximum algebra dotModel p s row a i.1=some m) :
    contribution algebra dotModel p s row a i=some
      (∑ j,mass p.length row (logicalScore p s μ i.1) m j*vAt p s μ (row+j.val) i.2.1.val) := by
  change sum 64 (fun j=>WithBot.realMul (probability algebra dotModel p s row a (i.1,j,()))
    (value algebra p s row (j,i.2.1,())))=_
  have hp : (fun j : Fin 64=>WithBot.realMul (probability algebra dotModel p s row a (i.1,j,()))
      (value algebra p s row (j,i.2.1,())))=
      (fun j=>some (mass p.length row (logicalScore p s μ i.1) m j*vAt p s μ (row+j.val) i.2.1.val)) := by
    funext j
    rw [probability_real p s μ h row i.1 a hi halign m hm j,value_real p s μ h row (j,i.2.1,())]
    by_cases hj : j.val<p.length-row <;> simp [mass,hj,hd]
  rw [hp,sum_some]

theorem accumulator_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [128,128]) (a : Acc R) (hi : s.pids 0*128+i.1.val<p.length)
    (hd : i.2.1.val<96) (halign : row%64=0) (m A : ℝ)
    (hm : Source.maximum algebra dotModel p s row a i.1=some m) (ha : a.value i=some A) :
    (update algebra dotModel p s row a).value i=some
      (A*alpha (a.maximum i.1) m+
        ∑ j,mass p.length row (logicalScore p s μ i.1) m j*vAt p s μ (row+j.val) i.2.1.val) := by
  change WithBot.realAdd (WithBot.realMul (a.value i) (rescale algebra dotModel p s row a i.1))
    (contribution algebra dotModel p s row a i)=_
  rw [ha,rescale_real p s row i.1 a m hm,contribution_real p s μ h row i a hi hd halign m hm]
  rfl

#axiomsClean maximum_finite
#axiomsClean normalizer_real
#axiomsClean accumulator_real
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Real
