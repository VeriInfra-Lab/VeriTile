import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.SourceSafety
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.IO

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.SourceSafety
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Backing for active logical windows, including optional state channels. -/
structure Windows (bounds : RegionName → Nat) (BH T D : Nat) (U S V : Bool) : Prop where
  rows : ∀ r, r="x" ∨ r="g" ∨ r="do" ∨ r="o" ∨ r="dx" ∨ r="dg" →
    ∀ bh, bh<BH → ∀ t, t<T → ∀ d, d<D → rowAddr bh T D t d<bounds r
  h0 : U=Bool.true → ∀ bh, bh<BH → ∀ d, d<D → stateAddr bh D d<bounds "h0"
  ht : S=Bool.true → ∀ bh, bh<BH → ∀ d, d<D → stateAddr bh D d<bounds "ht"
  dht : V=Bool.true → ∀ bh, bh<BH → ∀ d, d<D → stateAddr bh D d<bounds "dht"

private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (ha : Safety.traceSafeList M bounds a s) (ran : run M a s=some t)
    (hb : Safety.traceSafeList M bounds b t) : Safety.traceSafeList M bounds (a++b) s := by
  rw [Safety.traceSafeList_append]
  refine ⟨ha,?_⟩
  intro u hu
  have eq := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using hb

/-- The safety induction follows the actual loop successors supplied by the
existing execution invariant; no assumed output or successful run is added. -/
private theorem range_safe (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (stop : Nat) (P : Nat → State α → Prop)
    (one : ∀ i, i<stop → ∀ s, P i s → ∃ q,
      run M body (s.setReg idx .nat [] (fun _ => i))=some q ∧ P (i+1) q)
    (safe : ∀ i, i<stop → ∀ s, P i s →
      Safety.traceSafeList M bounds body (s.setReg idx .nat [] (fun _ => i)))
    (s : State α) (start : Nat) (bound : start≤stop) (initial : P start s) :
    Safety.rangeSafe M bounds idx start stop 1 body s := by
  have go : ∀ n i, i+n=stop → ∀ s, P i s → Safety.rangeSafe M bounds idx i stop 1 body s := by
    intro n
    induction n with
    | zero =>
      intro i hi s hs
      have eq : i=stop := by omega
      subst i
      rw [Safety.rangeSafe_unfold]
      simp
    | succ n ih =>
      intro i hi s hs
      have lt : i<stop := by omega
      obtain ⟨q,hq,hP⟩ := one i lt s hs
      rw [Safety.rangeSafe_unfold]
      simpa only [show ¬(1:Nat)=0 by omega,if_false,lt,if_true,hq] using
        And.intro (safe i lt s hs) (ih (i+1) (by omega) q hP)
  exact go (stop-start) start (by omega) s initial

theorem forward_safe (p : Ops α) (s : State α) (BH T D BD : Nat)
    (U S V : Bool) (bounds : RegionName → Nat) (W : Windows bounds BH T D U S V)
    (hb : s.pids 1<BH) : TraceSafe (algebra p) bounds (forward T D BD U S) s := by
  obtain ⟨q,he,hm,hp,hmask,hbh,hod,hx,hg,ho,hcarry⟩ := prologue p s T D BD U
  have P : FwdInv p s U T D BD 0 q :=
    ⟨hp,hmask,hbh,hod,hx,hg,ho,hcarry,by intro t ht;omega,by intro r a _;rw [hm]⟩
  have one := fun i (_ : i<T) q (h : FwdInv p s U T D BD i q) =>
    forward_invariant_step p s q U T D BD i h
  have safe : ∀ i, i<T → ∀ q, FwdInv p s U T D BD i q →
      Safety.traceSafeList (algebra p) bounds (forwardBody D BD)
        (q.setReg "_i" .nat [] (fun _ => i)) := by
    intro i hi q h
    let b := q.setReg "_i" .nat [] (fun _ => i)
    have pids : b.pids=s.pids := h.pids
    have lanes : lane b BD=lane s BD := by
      funext j
      simp [lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,pids]
    have rows : row b i T D BD=row s i T D BD := by
      funext j
      simp [row,lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
        VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,pids]
    apply forward_body_safe p b T D BD i (values p s U T D BD i)
      (by simpa [b,lanes] using h.mask) (by simpa [b,rows] using h.px)
      (by simpa [b,rows] using h.pg) (by simpa [b,rows] using h.po)
      (by simpa [b] using h.carry) bounds
    all_goals intro j hj
    · exact W.rows "x" (by simp) _ (pids ▸ hb) _ hi _ hj
    · exact W.rows "g" (by simp) _ (pids ▸ hb) _ hi _ hj
    · exact W.rows "o" (by simp) _ (pids ▸ hb) _ hi _ hj
  have hs := range_safe (algebra p) bounds "_i" (forwardBody D BD) T
    (FwdInv p s U T D BD) one safe q 0 (by omega) P
  obtain ⟨r,hr,hR⟩ := range_complete (algebra p) "_i" (forwardBody D BD) T
    (FwdInv p s U T D BD) one q 0 (by omega) P
  have ran : run (algebra p) [.forRange "_i" 0 T 1 (forwardBody D BD)] q=some r := by
    simpa [run,step] using hr
  have loop : Safety.traceSafeList (algebra p) bounds [.forRange "_i" 0 T 1 (forwardBody D BD)] q := by
    apply Safety.traceSafeList_cons_intro
    · simpa only [Safety.stepSafe_unfold] using hs
    · intro t _;exact Safety.traceSafeList_nil _ _ _
  have lanes : lane r BD=lane s BD := by
    funext j
    simp [lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,hR.pids]
  unfold TraceSafe
  rw [forward_split]
  apply append_safe _ _ _ _ _ _
    (append_safe _ _ _ _ _ _ (forward_prologue_safe p s T D BD U bounds
      (fun h j hj => W.h0 h _ hb _ hj)) he loop)
    (by rw [run_append,he];exact ran)
  exact forward_epilogue_safe p r T D BD S _
    (by simpa [lanes] using hR.mask) (by simpa [hR.pids] using hR.bh)
    (by simpa [lanes] using hR.channels) hR.carry bounds
    (fun h j hj => W.ht h _ (hR.pids ▸ hb) _ hj)

theorem backward_safe (p : Ops α) (s : State α) (BH T D BD : Nat)
    (U S V : Bool) (bounds : RegionName → Nat) (W : Windows bounds BH T D U S V)
    (hb : s.pids 1<BH) : TraceSafe (algebra p) bounds (backward T D BD U V) s := by
  obtain ⟨q,he,hm,hp,hmask,hbh,hod,hcarry⟩ := backward_prologue p s T D BD V
  have P : BwdInv p s U V T D BD 0 q :=
    ⟨hp,hmask,hbh,hod,hcarry,by intro t ht;omega,by intro t ht;omega,by intro r a _;rw [hm]⟩
  have one := fun i hi q (h : BwdInv p s U V T D BD i q) =>
    backward_invariant_step p s q U V T D BD i hi h
  have safe : ∀ i, i<T → ∀ q, BwdInv p s U V T D BD i q →
      Safety.traceSafeList (algebra p) bounds (backwardBody T D BD U)
        (q.setReg "step" .nat [] (fun _ => i)) := by
    intro i hi q h
    let b := q.setReg "step" .nat [] (fun _ => i)
    have pids : b.pids=s.pids := h.pids
    have lanes : lane b BD=lane s BD := by
      funext j
      simp [lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,pids]
    apply backward_body_safe p b T D BD i U (reverseValues p s V T D BD i)
      (by simp [b]) (by simpa [b,lanes] using h.mask)
      (by simpa [b,h.pids] using h.bh) (by simpa [b,lanes] using h.channels)
      (by simpa [b] using h.carry) bounds
    · exact fun j hj => W.rows "g" (by simp) _ (pids ▸ hb) _ (by omega) _ hj
    · exact fun j hj => W.rows "do" (by simp) _ (pids ▸ hb) _ (by omega) _ hj
    · exact fun j hj => W.rows "dx" (by simp) _ (pids ▸ hb) _ (by omega) _ hj
    · exact fun j hj => W.rows "dg" (by simp) _ (pids ▸ hb) _ (by omega) _ hj
    · exact fun ht j hj => W.rows "o" (by simp) _ (pids ▸ hb) _ (by omega) _ hj
    · exact fun hu j hj => W.h0 hu _ (pids ▸ hb) _ hj
  have hs := range_safe (algebra p) bounds "step" (backwardBody T D BD U) T
    (BwdInv p s U V T D BD) one safe q 0 (by omega) P
  unfold TraceSafe
  rw [backward_split]
  apply append_safe _ _ _ _ _ _
    (backward_prologue_safe p s T D BD V bounds (fun h j hj => W.dht h _ hb _ hj)) he
  apply Safety.traceSafeList_cons_intro
  · simpa only [Safety.stepSafe_unfold] using hs
  · intro t _;exact Safety.traceSafeList_nil _ _ _

#axiomsClean forward_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.SourceSafety
