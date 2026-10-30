import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealForwardStep
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source Mathematical
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def ForwardConsistent (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : FQuery p) (stop : Nat) (a : ForwardAccumulator R p) : Prop :=
  ∃ m,a.maximum (i.1,())=some m ∧
    a.denominator (i.1,())=some (mass (forwardQueryRow p s i.1) stop (logicalScore p s μ i.1) m) ∧
    a.numerator i=some
      (numerator (forwardQueryRow p s i.1) stop (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m /
       mass (forwardQueryRow p s i.1) stop (logicalScore p s μ i.1) m)

theorem initial_step (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (i : FQuery p) (hBN : 0<p.blockN)
    (hb : ∀ j : Fin p.blockN, forwardHeadRow p s+j.val<p.rows) :
    ForwardConsistent p s μ i p.blockN (forwardAdvance model p s 0 (forwardInitial model p)) := by
  obtain ⟨m,hm⟩:=maximum_finite p s μ h 0 i.1 (forwardInitial model p) hBN
    (by simpa using hb ⟨0,hBN⟩) (by simp)
  have hmass : mass (forwardQueryRow p s i.1) p.blockN (logicalScore p s μ i.1) m =
      blockMass (forwardQueryRow p s i.1) 0 p.blockN (logicalScore p s μ i.1) m := by
    simpa [mass] using mass_step (forwardQueryRow p s i.1) 0 p.blockN (logicalScore p s μ i.1) m
  have hnum : numerator (forwardQueryRow p s i.1) p.blockN (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m =
      blockNumerator (forwardQueryRow p s i.1) 0 p.blockN (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m := by
    simpa [numerator] using numerator_step (forwardQueryRow p s i.1) 0 p.blockN
      (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m
  have hn : forwardDenominator model p s 0 (forwardInitial model p) (i.1,())=
      some (mass (forwardQueryRow p s i.1) p.blockN (logicalScore p s μ i.1) m) := by
    rw [denominator_real p s μ h 0 i.1 (forwardInitial model p) (by simpa using hb) m 0 hm (by rfl)]
    simp [hmass]
  refine ⟨m,hm,hn,?_⟩
  rw [numerator_real p s μ h 0 i (forwardInitial model p) (by simpa using hb) m 0 _ 0 hm (by rfl) hn (by rfl)]
  simp [hnum]

theorem consistent_step (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (i : FQuery p) (hBN : 0<p.blockN)
    (c : Nat) (hc0 : 0<c) (a : ForwardAccumulator R p)
    (hb : ∀ j : Fin p.blockN, forwardHeadRow p s+c*p.blockN+j.val<p.rows)
    (hc : c*p.blockN ≤ forwardQueryRow p s i.1)
    (ha : ForwardConsistent p s μ i (c*p.blockN) a) :
    ForwardConsistent p s μ i ((c+1)*p.blockN) (forwardAdvance model p s c a) := by
  obtain ⟨old,ho,hl,hv⟩:=ha
  obtain ⟨m,hm⟩:=maximum_finite p s μ h c i.1 a hBN (by simpa using hb ⟨0,hBN⟩) hc
  let L:=blockMass (forwardQueryRow p s i.1) (c*p.blockN) p.blockN (logicalScore p s μ i.1) m+
    mass (forwardQueryRow p s i.1) (c*p.blockN) (logicalScore p s μ i.1) old*Real.exp (old-m)
  have hL : L=mass (forwardQueryRow p s i.1) ((c+1)*p.blockN) (logicalScore p s μ i.1) m := by
    dsimp [L]
    rw [mass_rescale,Nat.add_mul,Nat.one_mul,mass_step]
    ring
  have hn : forwardDenominator model p s c a (i.1,())=some L := by
    rw [denominator_real p s μ h c i.1 a hb m _ hm hl,ho]
    rfl
  refine ⟨m,hm,?_,?_⟩
  · exact hn.trans (congrArg some hL)
  · rw [numerator_real p s μ h c i a hb m _ L _ hm hl hn hv,ho]
    simp only [alpha]
    congr 1
    simpa only [Nat.add_mul,Nat.one_mul] using
      normalized_step (forwardQueryRow p s i.1) (c*p.blockN) p.blockN
        (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) old m (Nat.mul_pos hc0 hBN)

theorem loop_consistent (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (i : FQuery p) (hBN : 0<p.blockN)
    (n : Nat) (hn : 0<n)
    (hb : ∀ c<n, ∀ j : Fin p.blockN, forwardHeadRow p s+c*p.blockN+j.val<p.rows)
    (hc : ∀ c<n, c*p.blockN ≤ forwardQueryRow p s i.1) :
    ForwardConsistent p s μ i (n*p.blockN) (forwardRounds model p s n) := by
  induction n with
  | zero=>omega
  | succ n ih=>
    by_cases hz : n=0
    · subst n
      simpa [forwardRounds] using initial_step p s μ h i hBN (by simpa using hb 0 (by omega))
    · exact consistent_step p s μ h i hBN n (by omega) _ (hb n (by omega)) (hc n (by omega))
        (ih (by omega) (fun c h=>hb c (by omega)) (fun c h=>hc c (by omega)))

theorem answer_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (i : FQuery p) (hBN : 0<p.blockN)
    (hb : ∀ c<forwardBlocks p s, ∀ j : Fin p.blockN, forwardHeadRow p s+c*p.blockN+j.val<p.rows)
    (hc : ∀ c<forwardBlocks p s, c*p.blockN ≤ forwardQueryRow p s i.1)
    (hn : forwardQueryRow p s i.1+1 ≤ forwardBlocks p s*p.blockN) :
    (forwardRounds model p s (forwardBlocks p s)).numerator i=some
      (attention p.head p.scale (qAt p s μ) (kAt p s μ) (vAt p s μ)
        (forwardQueryRow p s i.1) i.2.1.val) := by
  have hn0 : 0<forwardBlocks p s := by
    by_contra h
    have hz : forwardBlocks p s=0 := by omega
    rw [hz,Nat.zero_mul] at hn
    omega
  obtain ⟨m,_,_,ha⟩:=loop_consistent p s μ h i hBN (forwardBlocks p s) hn0 hb hc
  rw [ha]
  congr 1
  exact normalized_done _ _ _ _ _ _ _ _ _ hn

#axiomsClean answer_real
#specNonCircular ForwardConsistent avoiding [FP.Structural.exec,forwardRounds,forwardWrite]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
