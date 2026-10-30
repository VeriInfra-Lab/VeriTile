import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem u_value (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (uInputs g)) (s : State α)
    (h : (uncachedIO rest ops g p q).InputBinding v s)
    (key odd : Bool) (i : ULane g) (live : uMask g p q key i) :
    Typed.uValue ops (Typed.uLayout g.request)
      ((uInvocation rest ops g p q).initial s) ((uInvocation rest ops g p q).initial s) key odd i=
      Cell.mk .real (rotation ops (uPayload g v key Bool.false i) (uPayload g v key Bool.true i)
        (uCoeff g v Bool.false i) (uCoeff g v Bool.true i) odd) := by
  unfold Typed.uValue
  simp only [u_offset g p q _ (u_pid0 rest ops g p q s) (u_pid1 rest ops g p q s),
    u_coefficient g q _ (u_pid1 rest ops g p q s),Invocation.initial_mem]
  change Cell.mk .real (rotation ops
    ((s.mem (region key) (uOffset g p q key Bool.false i)).read .real)
    ((s.mem (region key) (uOffset g p q key Bool.true i)).read .real)
    ((s.mem (coefficient Bool.false) (uCoefficient g q Bool.false i)).read .real)
    ((s.mem (coefficient Bool.true) (uCoefficient g q Bool.true i)).read .real) odd)=_
  rw [u_input rest ops g p q v s h key Bool.false i live,
    u_input rest ops g p q v s h key Bool.true i live,
    u_coeff_input rest ops g p q v s h Bool.false i live.2.1,
    u_coeff_input rest ops g p q v s h Bool.true i live.2.1]

theorem u_frame (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (uInputs g)) (s : State α) :
    (uncachedIO rest ops g p q).Frame v s
      (Typed.uTarget ops g.request ((uInvocation rest ops g p q).initial s)) := by
  let t := (uInvocation rest ops g p q).initial s
  have hp : t.pids 0=p := u_pid0 rest ops g p q s
  have hq : t.pids 1=q := u_pid1 rest ops g p q s
  intro r o outside
  have away (key : Bool) (leader : key=Bool.true → p%g.request.groups=0) :
      Pair.Outside (Typed.uRegion (Typed.uLayout g.request) key)
        (Typed.uActive (Typed.uLayout g.request) t key) (Typed.uOffset (Typed.uLayout g.request) t key) r o := by
    by_cases hr : r=Typed.uRegion (Typed.uLayout g.request) key
    · right
      intro odd i active
      have live : uMask g p q key i := (u_live g p q t hp hq key i).mp
        ⟨active,by simpa [hp] using leader⟩
      rw [u_offset g p q t hp hq]
      cases key <;> cases odd
      all_goals first
        | simpa [uncachedIO,uMemory,uOutputs,uInputs,hr,Typed.uRegion,Typed.uLayout] using outside.1 (0 : Fin 4) i live
        | simpa [uncachedIO,uMemory,uOutputs,uInputs,hr,Typed.uRegion,Typed.uLayout] using outside.1 (1 : Fin 4) i live
        | simpa [uncachedIO,uMemory,uOutputs,uInputs,hr,Typed.uRegion,Typed.uLayout] using outside.1 (2 : Fin 4) i live
        | simpa [uncachedIO,uMemory,uOutputs,uInputs,hr,Typed.uRegion,Typed.uLayout] using outside.1 (3 : Fin 4) i live
    · exact Or.inl hr
  change (Typed.uTarget ops g.request t).mem r o=s.mem r o
  unfold Typed.uTarget
  by_cases lead : t.pids 0%g.request.groups=0
  · rw [if_pos lead]
    unfold Typed.uBranch
    rw [Pair.frame _ _ _ _ _ _ _ (away Bool.true (by simpa [hp] using lead)),
      Pair.frame _ _ _ _ _ _ _ (away Bool.false (by simp))]
    exact Invocation.initial_mem _ _ |> fun hmem => congrFun (congrFun hmem r) o
  · rw [if_neg lead]
    unfold Typed.uBranch
    rw [Pair.frame _ _ _ _ _ _ _ (away Bool.false (by simp))]
    exact Invocation.initial_mem _ _ |> fun hmem => congrFun (congrFun hmem r) o

theorem uncached_implements (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (heads : ∀ key,0<g.request.heads key) :
    (uncachedIO rest ops g p q).Implements (uTarget ops g) := by
  apply ProgramIO.Implements.intro
  · intro v bounds hb s _
    exact (scheduleSafe_singleton _ _ _).mpr (u_safe rest ops g p q v bounds hb s)
  · intro v s h
    let t := (uInvocation rest ops g p q).initial s
    have hp : t.pids 0=p := u_pid0 rest ops g p q s
    have hq : t.pids 1=q := u_pid1 rest ops g p q s
    obtain ⟨final,run,mem,outputs,_⟩ := Typed.uncached_raw_successful rest ops g.request heads t
    refine ⟨final,?_,?_,?_⟩
    · exact (runSchedule_singleton _ _).trans run
    · intro o i live
      have value (key odd : Bool) (i : ULane g) (hi : uMask g p q key i) :
          final.mem (region key) (uOffset g p q key odd i)=
            .mk .real (rotation ops (uPayload g v key Bool.false i) (uPayload g v key Bool.true i)
              (uCoeff g v Bool.false i) (uCoeff g v Bool.true i) odd) := by
        have result := outputs key odd i ((u_live g p q t hp hq key i).mpr hi)
        rw [u_offset g p q t hp hq,u_value rest ops g p q v s h key odd i hi] at result
        exact result
      change Fin 4 at o
      fin_cases o
      · change mapCell id (final.mem (region Bool.false) (uOffset g p q Bool.false Bool.false i)) = _
        rw [value Bool.false Bool.false i live]
        rfl
      · change mapCell id (final.mem (region Bool.false) (uOffset g p q Bool.false Bool.true i)) = _
        rw [value Bool.false Bool.true i live]
        rfl
      · change mapCell id (final.mem (region Bool.true) (uOffset g p q Bool.true Bool.false i)) = _
        rw [value Bool.true Bool.false i live]
        rfl
      · change mapCell id (final.mem (region Bool.true) (uOffset g p q Bool.true Bool.true i)) = _
        rw [value Bool.true Bool.true i live]
        rfl
    · intro r o outside
      rw [mem]
      exact u_frame rest ops g p q v s r o outside

#axiomsClean uncached_implements
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
