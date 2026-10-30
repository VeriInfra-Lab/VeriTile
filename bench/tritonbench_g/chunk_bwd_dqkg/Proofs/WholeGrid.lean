import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.WholeStep

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

/-- Each kernel call executes against the memory returned by the previous call. -/
def runGrid (c : Config) : List (CTA c) → BlockState → Option BlockState
  | [], s => some s
  | a :: rest, s => do
      let t ← exec (kernel c).eraseDType.toAlgKernel (prepare c s a)
      runGrid c rest t

def gridWrites (c : Config) (order : List (CTA c)) : WriteFootprint :=
  fun addr => ∃ a ∈ order, writes c a addr

def allCTAs (c : Config) : List (CTA c) :=
  (List.finRange c.NK).flatMap fun k =>
    (List.finRange c.NT).flatMap fun t =>
      (List.finRange c.heads).map fun h => ⟨k,t,h⟩

theorem mem_allCTAs (c : Config) (a : CTA c) : a ∈ allCTAs c := by
  cases a with
  | mk k t h => simp [allCTAs,List.mem_flatMap,List.mem_map]

theorem read_of_frame (s t : BlockState) (w : WriteFootprint)
    (hf : BlockState.WriteWithin w s t) (region : RegionName) (off : Nat)
    (hm : ¬w (region,off)) : t.readMem region off = s.readMem region off := by
  simp only [BlockState.readMem]
  rw [← hf region off hm]

theorem inputs_of_step_frame (c : Config) (a : CTA c) (s t : BlockState)
    (hf : BlockState.WriteWithin (writes c a) s t) : inputs t = inputs s := by
  have hr (region : RegionName) (h : region ∈ (["q","k","v","h","g","do","dh"] : List RegionName)) :
      t.readMem region = s.readMem region := by
    funext off
    exact read_of_frame s t _ hf region off (input_unwritten c a region off h)
  simp only [inputs,RealPost.inputs,
    hr "q" (by simp),hr "k" (by simp),hr "v" (by simp),hr "h" (by simp),
    hr "g" (by simp),hr "do" (by simp),hr "dh" (by simp)]

theorem unwritten_by_others (c : Config) (a : CTA c) (order : List (CTA c))
    (hm : a ∉ order) (region : RegionName) (off : Nat)
    (ha : writes c a (region,off)) : ¬gridWrites c order (region,off) := by
  rintro ⟨b,hb,hw⟩
  have eq := writes_unique c b a region off hw ha
  subst b
  exact hm hb

/-- Whole execution in arbitrary CTA-list order. The proof derives unchanged
input reads from each predecessor's exact frame, then preserves outputs from
other CTAs using address uniqueness. Repeated CTAs also compute the same value. -/
theorem grid_exists (c : Config) (order : List (CTA c)) (s : BlockState) :
    ∃ final, runGrid c order s = some final ∧
      (∀ a ∈ order, ∀ r e, activeQK c a r e → final.readMem "dq" (qAddress c a r e) =
        RealMath.dq c.layout a.launch (inputs s) r e) ∧
      (∀ a ∈ order, ∀ r e, activeQK c a r e → final.readMem "dk" (qAddress c a r e) =
        RealMath.dk c.layout a.launch (inputs s) r e) ∧
      (∀ a ∈ order, ∀ r, activeG c a r → final.readMem "dg" (gAddress c a r) =
        RealMath.dg c.layout a.launch (inputs s) r) ∧
      BlockState.WriteWithin (gridWrites c order) s final := by
  classical
  induction order generalizing s with
  | nil =>
    exact ⟨s,rfl,by simp,by simp,by simp,by intro r o _; rfl⟩
  | cons a rest ih =>
    obtain ⟨t,he,hq,hk,hg,hf⟩ := step_exists c s a
    obtain ⟨final,hrun,fq,fk,fg,ff⟩ := ih t
    have hin := inputs_of_step_frame c a s t hf
    refine ⟨final,?_,?_,?_,?_,?_⟩
    · simp only [runGrid,he,bind,Option.bind,hrun]
    · intro b hb r e ha
      by_cases hbr : b ∈ rest
      · simpa only [hin] using fq b hbr r e ha
      · have hba : b = a := (List.mem_cons.mp hb).resolve_right hbr
        subst b
        have hw : writes c a ("dq",qAddress c a r e) :=
          (writes_cases c a _ _).2 (Or.inl ⟨rfl,r,e,ha,rfl⟩)
        exact (read_of_frame t final _ ff _ _
          (unwritten_by_others c a rest hbr _ _ hw)).trans (hq r e ha)
    · intro b hb r e ha
      by_cases hbr : b ∈ rest
      · simpa only [hin] using fk b hbr r e ha
      · have hba : b = a := (List.mem_cons.mp hb).resolve_right hbr
        subst b
        have hw : writes c a ("dk",qAddress c a r e) :=
          (writes_cases c a _ _).2 (Or.inr (Or.inl ⟨rfl,r,e,ha,rfl⟩))
        exact (read_of_frame t final _ ff _ _
          (unwritten_by_others c a rest hbr _ _ hw)).trans (hk r e ha)
    · intro b hb r ha
      by_cases hbr : b ∈ rest
      · simpa only [hin] using fg b hbr r ha
      · have hba : b = a := (List.mem_cons.mp hb).resolve_right hbr
        subst b
        have hw : writes c a ("dg",gAddress c a r) :=
          (writes_cases c a _ _).2 (Or.inr (Or.inr ⟨rfl,r,ha,rfl⟩))
        exact (read_of_frame t final _ ff _ _
          (unwritten_by_others c a rest hbr _ _ hw)).trans (hg r ha)
    · intro region off hm
      have hma : ¬writes c a (region,off) := fun h => hm ⟨a,by simp,h⟩
      have hmr : ¬gridWrites c rest (region,off) := by
        rintro ⟨b,hb,hw⟩
        exact hm ⟨b,List.mem_cons_of_mem a hb,hw⟩
      exact (hf region off hma).trans (ff region off hmr)

def expectedQ (c : Config) (s : BlockState) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : ℝ :=
  RealMath.dq c.layout (qOwner c h t e).launch (inputs s) (rowLane t.val) (keyLane c e.val)
def expectedK (c : Config) (s : BlockState) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : ℝ :=
  RealMath.dk c.layout (qOwner c h t e).launch (inputs s) (rowLane t.val) (keyLane c e.val)
def expectedG (c : Config) (s : BlockState) (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) : ℝ :=
  RealMath.dg c.layout (gOwner c k h t).launch (inputs s) (rowLane t.val)

/-- Every logical returned cell is covered, including short time/key/value
tiles. Empty grids execute no kernel and leave memory unchanged. -/
theorem whole_result (c : Config) (s : BlockState) (order : List (CTA c))
    (horder : order.Perm (allCTAs c)) :
    ∃ final, runGrid c order s = some final ∧
      (∀ h t e, final.readMem "dq" ((h.val*c.T+t.val)*c.K+e.val) = expectedQ c s h t e) ∧
      (∀ h t e, final.readMem "dk" ((h.val*c.T+t.val)*c.K+e.val) = expectedK c s h t e) ∧
      (∀ k h t, final.readMem "dg" ((k.val*c.heads+h.val)*c.T+t.val) = expectedG c s k h t) ∧
      BlockState.WriteWithin (gridWrites c (allCTAs c)) s final := by
  obtain ⟨final,he,hq,hk,hg,hf⟩ := grid_exists c order s
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
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
