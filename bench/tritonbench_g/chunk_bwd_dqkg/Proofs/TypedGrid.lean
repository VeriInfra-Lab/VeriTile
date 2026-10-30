import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedInputFrame
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.WholeGrid
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- Every CTA executes the actual typed source against its predecessor's memory. -/
def runGrid (M : Algebra α) (dot : Dot α) (c : Config) : List (CTA c) → State α → Option (State α)
  | [],s=>some s
  | a::rest,s=>do
      let t←FP.Structural.exec (engine (withDot M dot)) (kernel c) (prepare c s a)
      runGrid M dot c rest t

/-- Whole execution in arbitrary CTA-list order. The proof derives unchanged
input reads from each predecessor's exact frame, then preserves outputs from
other CTAs using address uniqueness. Repeated CTAs also compute the same value. -/
theorem grid_exists (M : Algebra α) (dot : Dot α) (c : Config) (order : List (CTA c)) (s : State α) :
    ∃ final, runGrid M dot c order s = some final ∧
      (∀ a ∈ order, ∀ r e, activeQK c a r e → final.mem "dq" (qAddress c a r e) =
        .mk .fp32 (resultQ M dot c s a r e)) ∧
      (∀ a ∈ order, ∀ r e, activeQK c a r e → final.mem "dk" (qAddress c a r e) =
        .mk .fp32 (resultK M dot c s a r e)) ∧
      (∀ a ∈ order, ∀ r, activeG c a r → final.mem "dg" (gAddress c a r) =
        .mk .real (resultG M dot c s a r)) ∧
      (∀ region off,¬gridWrites c order (region,off) → final.mem region off=s.mem region off) := by
  classical
  induction order generalizing s with
  | nil =>
    exact ⟨s,rfl,by simp,by simp,by simp,by intro r o _; rfl⟩
  | cons a rest ih =>
    let t := finalState M dot c (prepare c s a)
    have he := source_runs M dot c (prepare c s a)
    have hq := tile_q M dot c s a
    have hk := tile_k M dot c s a
    have hg := tile_g M dot c s a
    have hf := tile_frame M dot c s a
    obtain ⟨final,hrun,fq,fk,fg,ff⟩ := ih t
    have hin (b : CTA c) := results_inputs M dot c t s b (tile_inputs M dot c s a)
    refine ⟨final,?_,?_,?_,?_,?_⟩
    · simp only [runGrid,he,bind,Option.bind]
      exact hrun
    · intro b hb r e ha
      by_cases hbr : b ∈ rest
      · simpa only [(hin b).1] using fq b hbr r e ha
      · have hba : b = a := (List.mem_cons.mp hb).resolve_right hbr
        subst b
        have hw : writes c a ("dq",qAddress c a r e) :=
          (writes_cases c a _ _).2 (Or.inl ⟨rfl,r,e,ha,rfl⟩)
        exact (ff _ _ (unwritten_by_others c a rest hbr _ _ hw)).trans (hq r e ha)
    · intro b hb r e ha
      by_cases hbr : b ∈ rest
      · simpa only [(hin b).2.1] using fk b hbr r e ha
      · have hba : b = a := (List.mem_cons.mp hb).resolve_right hbr
        subst b
        have hw : writes c a ("dk",qAddress c a r e) :=
          (writes_cases c a _ _).2 (Or.inr (Or.inl ⟨rfl,r,e,ha,rfl⟩))
        exact (ff _ _ (unwritten_by_others c a rest hbr _ _ hw)).trans (hk r e ha)
    · intro b hb r ha
      by_cases hbr : b ∈ rest
      · simpa only [(hin b).2.2] using fg b hbr r ha
      · have hba : b = a := (List.mem_cons.mp hb).resolve_right hbr
        subst b
        have hw : writes c a ("dg",gAddress c a r) :=
          (writes_cases c a _ _).2 (Or.inr (Or.inr ⟨rfl,r,ha,rfl⟩))
        exact (ff _ _ (unwritten_by_others c a rest hbr _ _ hw)).trans (hg r ha)
    · intro region off hm
      have hma : ¬writes c a (region,off) := fun h => hm ⟨a,by simp,h⟩
      have hmr : ¬gridWrites c rest (region,off) := by
        rintro ⟨b,hb,hw⟩
        exact hm ⟨b,List.mem_cons_of_mem a hb,hw⟩
      exact (ff region off hmr).trans (hf region off hma)

def expectedQ (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : α :=
  resultQ M dot c s (qOwner c h t e) (rowLane t.val) (keyLane c e.val)
def expectedK (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : α :=
  resultK M dot c s (qOwner c h t e) (rowLane t.val) (keyLane c e.val)
def expectedG (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) : α :=
  resultG M dot c s (gOwner c k h t) (rowLane t.val)

/-- Every logical returned cell is covered, including short time/key/value
tiles. Empty grids execute no kernel and leave memory unchanged. -/
theorem whole_result (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (order : List (CTA c))
    (horder : order.Perm (allCTAs c)) :
    ∃ final, runGrid M dot c order s = some final ∧
      (∀ h t e, final.mem "dq" ((h.val*c.T+t.val)*c.K+e.val) = .mk .fp32 (expectedQ M dot c s h t e)) ∧
      (∀ h t e, final.mem "dk" ((h.val*c.T+t.val)*c.K+e.val) = .mk .fp32 (expectedK M dot c s h t e)) ∧
      (∀ k h t, final.mem "dg" ((k.val*c.heads+h.val)*c.T+t.val) = .mk .real (expectedG M dot c s k h t)) ∧
      (∀ region off,¬gridWrites c (allCTAs c) (region,off) → final.mem region off=s.mem region off) := by
  obtain ⟨final,he,hq,hk,hg,hf⟩ := grid_exists M dot c order s
  have hm (a : CTA c) : a ∈ order := horder.mem_iff.mpr (mem_allCTAs c a)
  refine ⟨final,he,?_,?_,?_,?_⟩
  · intro h t e
    simpa only [q_owner_address,expectedQ] using
      hq (qOwner c h t e) (hm _) (rowLane t.val) (keyLane c e.val) (q_owner_active c h t e)
  · intro h t e
    simpa only [q_owner_address,expectedK] using
      hk (qOwner c h t e) (hm _) (rowLane t.val) (keyLane c e.val) (q_owner_active c h t e)
  · intro k h t
    simpa only [g_owner_address,expectedG] using
      hg (gOwner c k h t) (hm _) (rowLane t.val) (g_owner_active c k h t)
  · intro region off hn
    apply hf region off
    rintro ⟨a,_,ha⟩
    exact hn ⟨a,mem_allCTAs c a,ha⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
