import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.RegionProjection

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReadProjection
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def Outside (p : Problem) (b : Buffers) (r : RegionName) (o : Nat) : Prop :=
  r≠"C" ∨ ∀ i, i<p.M → ∀ j, j<p.N → o≠i*b.C.rowStride+j*b.C.columnStride

def Tagged (s : State α) (o : Nat) : Prop := ∃ v, s.mem "C" o=Cell.mk .fp16 v

def Covered (p : Problem) (b : Buffers) (ids : List Nat) (o : Nat) : Prop :=
  ∃ pid∈ids, ∃ lane : Source.Lane, Source.active p pid lane ∧
    o=p.row pid lane.1.val*b.C.rowStride+p.column pid lane.2.val*b.C.columnStride

theorem program_tags (M : Algebra α) (ops : Operations α) (p : Problem) (b : Buffers)
    (L : Layout p b) (s t : State α) (ht : s.pids 0<p.tiles)
    (ran : FP.Structural.exec (backend M (selectedDot ops)) (source p b) s=some t) :
    ∀ o, Tagged s o ∨ Covered p b [s.pids 0] o → Tagged t o := by
  obtain ⟨u,hu,_,value,frame⟩ := programExecution M (selectedDot ops) p b s (active_injective p b L s ht)
  have eq : u=t := Option.some.inj (hu.symm.trans ran)
  subst u
  intro o h
  by_cases written : ∃ i, active p s i ∧ o=cOffset p b s i
  · obtain ⟨i,hi,rfl⟩ := written
    exact ⟨_,value i hi⟩
  · have outside : ∀ i, active p s i → o≠cOffset p b s i := by
      intro i hi eq;exact written ⟨i,hi,eq⟩
    rcases h with ⟨v,hv⟩ | ⟨pid,hp,lane,hl,he⟩
    · exact ⟨v,(frame "C" o (Or.inr outside)).trans hv⟩
    · have pid : pid=s.pids 0 := List.mem_singleton.mp hp
      subst pid
      let i : CIndex := (lane.1,lane.2,())
      have hi : active p s i := (active_eq p s i ht).mpr hl
      exact False.elim (outside i hi (he.trans (cOffset_eq p b s i ht hi).symm))

theorem outside_program (p : Problem) (b : Buffers) (s : State α)
    (ht : s.pids 0<p.tiles) (r : RegionName) (o : Nat) (h : Outside p b r o) :
    r≠"C" ∨ ∀ i, active p s i → o≠cOffset p b s i := by
  rcases h with h | h
  · exact Or.inl h
  · right
    intro i hi
    rw [cOffset_eq p b s i ht hi]
    have a := (active_eq p s i ht).mp hi
    exact h _ a.1 _ a.2

/-- Complete genuine source schedule, with its exact raw frame and a separate
FP16 tag-preservation/coverage conclusion before decoded output observation. -/
theorem schedule (M : Algebra α) (ops : Operations α) (law : ScalarLaws M ops)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<g.problem.tiles) (s : State α) :
    ∃ final, runSchedule (ids.map (IO.invocation M ops g b)) s=some final ∧
      decode (b.views g) final=CrossEntropy1.Schedule.run
        (Source.step (sourceArithmetic ops) g.problem (views (b.views g))) ids (decode (b.views g) s) ∧
      (∀ o, Tagged s o ∨ Covered g.problem (b.views g) ids o → Tagged final o) ∧
      (∀ r o, Outside g.problem (b.views g) r o → final.mem r o=s.mem r o) := by
  induction ids generalizing s with
  | nil =>
    refine ⟨s,rfl,rfl,?_,?_⟩
    · intro o h
      rcases h with h | ⟨pid,hp,_⟩
      · exact h
      · simp at hp
    · intro r o _;rfl
  | cons pid tail ih =>
    let start := (IO.invocation M ops g b pid).initial s
    have hp : start.pids 0=pid := by simp [start,Invocation.initial,IO.invocation,bindRegisters]
    have ht : start.pids 0<g.problem.tiles := hp ▸ valid pid (by simp)
    obtain ⟨next,ran,proj,frame⟩ := program M ops law g.problem (b.views g) L start ht
    obtain ⟨final,hr,hf,tags,ff⟩ := ih (fun pid hp => valid pid (by simp [hp])) next
    refine ⟨final,?_,?_,?_,?_⟩
    · change (Invocation.run (IO.invocation M ops g b pid) s).bind
        (runSchedule (tail.map (IO.invocation M ops g b)))=some final
      rw [show Invocation.run (IO.invocation M ops g b pid) s=some next from ran]
      exact hr
    · rw [hf,proj,hp]
      rfl
    · intro o h
      apply tags o
      rcases h with old | ⟨tile,hmem,lane,ha,he⟩
      · exact Or.inl (program_tags M ops g.problem (b.views g) L start next ht ran o (Or.inl old))
      · rcases List.mem_cons.mp hmem with eq | hmem
        · subst tile
          exact Or.inl (program_tags M ops g.problem (b.views g) L start next ht ran o
            (Or.inr ⟨pid,by simp [hp],lane,ha,he⟩))
        · exact Or.inr ⟨tile,hmem,lane,ha,he⟩
    · intro r o ho
      exact (ff r o ho).trans (frame r o (outside_program g.problem (b.views g) start ht r o ho))

#axiomsClean schedule
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReadProjection
