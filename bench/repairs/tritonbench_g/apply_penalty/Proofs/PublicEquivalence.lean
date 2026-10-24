import bench.repairs.tritonbench_g.apply_penalty.Proofs.AccessSafety
import bench.repairs.tritonbench_g.apply_penalty.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.ApplyPenalty.PublicIO
open VeriTile.Triton FP.Structural FP.Guarded
noncomputable section
variable {α : Type} [Inhabited α]

private theorem start_binding (M : Algebra α) (stride B pid : Nat)
    (xs : Inputs α B) (s : State α) (hp : s.pids 0=pid)
    (hin : (io M stride B pid).InputBinding xs s) :
    Rewrite.start s=xs 3 PUnit.unit := by
  simpa [Rewrite.start,hp,io,inputs,mapValue] using hin 3 PUnit.unit trivial

private theorem stop_binding (M : Algebra α) (stride B pid : Nat)
    (xs : Inputs α B) (s : State α) (hp : s.pids 0=pid)
    (hin : (io M stride B pid).InputBinding xs s) :
    Rewrite.stop s=xs 4 PUnit.unit := by
  simpa [Rewrite.stop,hp,io,inputs,mapValue] using hin 4 PUnit.unit trivial

private theorem active_binding (M : Algebra α) (stride B pid : Nat)
    (xs : Inputs α B) (s : State α) (hp : s.pids 0=pid)
    (hin : (io M stride B pid).InputBinding xs s) (j : TileIndex [B]) :
    Rewrite.active s j ↔ active xs j := by
  simp only [Rewrite.active,active,start_binding M stride B pid xs s hp hin,
    stop_binding M stride B pid xs s hp hin]
  rfl

private theorem address_binding (M : Algebra α) (stride B pid : Nat)
    (xs : Inputs α B) (s : State α) (hp : s.pids 0=pid)
    (hin : (io M stride B pid).InputBinding xs s) (j : TileIndex [B])
    (hj : Rewrite.active s j) :
    Rewrite.address stride s j=address stride pid xs j := by
  have ha := (active_binding M stride B pid xs s hp hin j).mp hj
  have hi : (s.mem "ids" (Nat.add (xs 3 PUnit.unit) j.1.val)).read .nat=xs 5 j :=
    hin 5 j ha
  simp only [Rewrite.address,hj,↓reduceIte,hp,start_binding M stride B pid xs s hp hin,
    address]
  exact congrArg (fun t : Nat => pid*stride+t) hi

/-- The logical domain is precisely the old loaded-operand domain, transported
through input bindings; no source execution is required as a premise. -/
theorem division_domain (M : Algebra α) (D : Domain α) (lt : α→α→Bool)
    (stride B pid : Nat) (xs : Inputs α B) (s : State α) (hp : s.pids 0=pid)
    (hin : (io M stride B pid).InputBinding xs s) (hd : DivisionDomain M D lt xs) :
    Rewrite.DivisionDomain M D lt stride B s := by
  intro j hj positive
  have ha := (active_binding M stride B pid xs s hp hin j).mp hj
  have hx : (s.mem "logits" (address stride pid xs j)).read .real=xs 7 j := hin 7 j ha
  have hr : (s.mem "repetition" pid).read .real=xs 2 PUnit.unit := hin 2 PUnit.unit trivial
  have load : Rewrite.loadedLogit M stride s j=M.fp32Load (xs 7 j) := by
    simp only [Rewrite.loadedLogit,hj,↓reduceIte,address_binding M stride B pid xs s hp hin j hj,hx]
  have rep : Rewrite.loadedRepetition M s=M.fp32Load (xs 2 PUnit.unit) := by
    simp only [Rewrite.loadedRepetition,hp,hr]
  rw [load,rep]
  exact hd j ha (by simpa only [load] using positive)

theorem source_safe (M : Algebra α) (lt : α→α→Bool)
    (hlt : M.compareLt (some .fp32) .real=some lt)
    (variant : Bool) (stride B pid : Nat) (xs : Inputs α B)
    (bounds : RegionName→Nat) (hb : (io M stride B pid).InBounds bounds xs)
    (s : State α) (hp : s.pids 0=pid) (hin : (io M stride B pid).InputBinding xs s) :
    TraceSafe (FP.ProfiledRewrite.engine M) bounds
      (if variant then Rewrite.reciprocalSource stride B else Rewrite.originalSource stride B) s := by
  apply AccessSafety.source M lt hlt variant stride B s bounds
  · simpa only [hp] using hb.2.1 0 PUnit.unit trivial
  · simpa only [hp] using hb.2.1 1 PUnit.unit trivial
  · simpa only [hp] using hb.2.1 2 PUnit.unit trivial
  · simpa only [hp] using hb.2.1 3 PUnit.unit trivial
  · simpa only [hp] using hb.2.1 4 PUnit.unit trivial
  · intro j hj
    rw [start_binding M stride B pid xs s hp hin]
    exact hb.2.1 5 j ((active_binding M stride B pid xs s hp hin j).mp hj)
  · intro j hj
    rw [start_binding M stride B pid xs s hp hin]
    exact hb.2.1 6 j ((active_binding M stride B pid xs s hp hin j).mp hj)
  · intro j hj
    rw [address_binding M stride B pid xs s hp hin j hj]
    exact hb.2.1 7 j ((active_binding M stride B pid xs s hp hin j).mp hj)

private theorem frame (M : Algebra α) (lt : α→α→Bool)
    (hlt : M.compareLt (some .fp32) .real=some lt)
    (stride B pid : Nat) (xs : Inputs α B) (s : State α) (hp : s.pids 0=pid)
    (hin : (io M stride B pid).InputBinding xs s) :
    (io M stride B pid).Frame xs s (Rewrite.originalExecution M lt hlt stride B s).val := by
  intro r offset outside
  apply Rewrite.original_frame M lt hlt stride B s r offset
  by_cases hr : r="logits"
  · right
    intro j hj
    have ha := (active_binding M stride B pid xs s hp hin j).mp hj
    have h := outside.1 0 j ha
    change r≠"logits" ∨ offset≠address stride pid xs j at h
    rw [address_binding M stride B pid xs s hp hin j hj]
    exact h.resolve_left (not_not.mpr hr)
  · exact Or.inl hr

/-- Reuse complete source execution and full-cell equality; connect them to
the shared descriptor's actual launches, read windows, trace safety and frame. -/
@[spec_rule] theorem equivalent (R : FP.ScalarArithmetic.Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (lt : α→α→Bool)
    (hlt : M.compareLt (some .fp32) .real=some lt) (stride B pid : Nat) :
    (io M stride B pid).EquivalentOn (candidate M stride B pid) (DivisionDomain M D lt) := by
  intro xs hd bounds hb initial hin
  let s := (launch M Bool.false stride B pid).initial initial
  have hp : s.pids 0=pid := rfl
  have hs : s.mem=initial.mem := Invocation.initial_mem _ _
  have binding : (io M stride B pid).InputBinding xs s := by
    intro i j hj
    rw [hs]
    exact hin i j hj
  have safe (v : Bool) : (launch M v stride B pid).Safe bounds initial :=
    source_safe M lt hlt v stride B pid xs bounds hb s hp binding
  have domain := division_domain M D lt stride B pid xs s hp binding hd
  have hm := Rewrite.reciprocal_memory R M D hM lt hlt stride B s domain
  have fr := frame M lt hlt stride B pid xs s hp binding
  refine ⟨?_,?_,(Rewrite.originalExecution M lt hlt stride B s).val,
    (Rewrite.reciprocalExecution M lt hlt stride B s).val,?_,?_,?_,?_,?_⟩
  · exact (scheduleSafe_singleton _ _ _).mpr (safe Bool.false)
  · exact (scheduleSafe_singleton _ _ _).mpr (safe Bool.true)
  · simpa only [io,runSchedule_singleton,Invocation.run,launch] using
      (Rewrite.originalExecution M lt hlt stride B s).property
  · simpa only [candidate,runSchedule_singleton,Invocation.run,launch] using
      (Rewrite.reciprocalExecution M lt hlt stride B s).property
  · intro o j _
    change mapCell id ((Rewrite.originalExecution M lt hlt stride B s).val.mem _ _)=
      mapCell id ((Rewrite.reciprocalExecution M lt hlt stride B s).val.mem _ _)
    rw [hm]
  · intro r offset outside
    exact (fr r offset outside).trans (congrFun (congrFun hs r) offset)
  · intro r offset outside
    rw [← hm]
    exact (fr r offset outside).trans (congrFun (congrFun hs r) offset)

#axiomsClean division_domain
#axiomsClean source_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ApplyPenalty.PublicIO
