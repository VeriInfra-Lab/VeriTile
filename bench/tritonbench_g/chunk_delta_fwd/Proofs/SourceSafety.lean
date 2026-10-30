import bench.tritonbench_g.chunk_delta_fwd.Proofs.SourceSafetyInner

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem outer_body (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current) (hh : HasH p frozen current)
    (bounds : RegionName → Nat) (hb : Bounds p pid bounds) (hc : c < p.NT) :
    traceSafeList (engine (withDot M D)) bounds (outer p).body
      (current.setReg "i_t" .nat [] (fun _ => c)) := by
  rw [inner_parts,List.append_assoc,traceSafeList_append]
  refine ⟨history (engine (withDot M D)) p pid input current c frozen ctx hh bounds hb hc,?_⟩
  intro start ran
  let e := historyExecution M D p pid input current c frozen ctx hh
  have eq : start = e.val := Option.some.inj (ran.symm.trans e.property)
  subst start
  have hv := history_values M D p pid input current c frozen ctx hh
  rw [traceSafeList_append]
  refine ⟨?_,fun s _ => update (engine (withDot M D)) p s bounds⟩
  rw [traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · have h0 : (inner p).start = .constNat 0 := rfl
    have hn : (inner p).stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat p.BT) (.constNat p.BC)) (.constNat 1))
        (.constNat p.BC) := rfl
    have h1 : (inner p).stride = .constNat 1 := rfl
    simp only [stepSafe_unfold,h0,hn,h1,safeOp_unfold,evalOp_unfold,Option.bind_some,
      true_and,and_self,numeric,bop]
    exact inner_range M D p pid input e.val c frozen hv.1 hv.2.1 hv.2.2.1 hv.2.2.2 bounds hb
  · cases step (engine (withDot M D)) _ e.val <;> simp only [traceSafeList_nil]

theorem outer_range (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α)
    (ctx : Ready pid input current) (hh : HasH p (initial M p pid input) current)
    (bounds : RegionName → Nat) (hb : Bounds p pid bounds) :
    rangeSafe (engine (withDot M D)) bounds "i_t" 0 p.NT 1 (outer p).body current := by
  let I := fun n (s : State α) => Ready pid input s ∧ HasH p (states M D p pid input n) s
  apply rangeSafe_of_invariant (engine (withDot M D)) bounds "i_t" p.NT 1 (outer p).body I
  · intro n hn s inv
    exact outer_body M D p pid input s n _ inv.1 inv.2 bounds hb hn
  · intro n _ s inv t ran
    obtain ⟨u,he,hctx,hh,_⟩ := outerStep M D p pid input s n
      (states M D p pid input n) inv.1 inv.2
    rw [ran] at he
    cases he
    exact ⟨hctx,hh⟩
  · exact ⟨ctx,hh⟩

/-- Safety is proved along the actual complete quoted source, including both
nested ranges and both optional descriptor accesses. -/
theorem source (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (s : State α) (hk : s.pids 0 = 0) (bounds : RegionName → Nat)
    (hb : Bounds p (s.pids 1,s.pids 2) bounds) :
    TraceSafe (engine (withDot M D)) bounds (kernel p) s := by
  unfold TraceSafe
  rw [outer_parts,List.append_assoc,traceSafeList_append]
  refine ⟨entry (engine (withDot M D)) p s hk bounds hb,?_⟩
  intro start ran
  have he := entry_run M D p s
  rw [ran] at he
  cases he
  have hv := entry_ready M D p s hk
  rw [traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [traceSafeList_unfold]
    refine ⟨?_,?_⟩
    · simp only [stepSafe_unfold]
      exact outer_range M D p (s.pids 1,s.pids 2) s.mem (entryState M p s)
        hv.1 hv.2 bounds hb
    · cases step (engine (withDot M D)) _ (entryState M p s) <;> simp only [traceSafeList_nil]
  · intro last hlast
    obtain ⟨t,ht,hctx,hh,_⟩ := outerLoop M D p (s.pids 1,s.pids 2) s.mem
      (entryState M p s) hv.1 hv.2
    simp [run,step,Option.bind,ht] at hlast
    cases hlast
    exact finish (engine (withDot M D)) p (s.pids 1,s.pids 2) s.mem last _ hctx hh bounds hb

#axiomsClean source
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
