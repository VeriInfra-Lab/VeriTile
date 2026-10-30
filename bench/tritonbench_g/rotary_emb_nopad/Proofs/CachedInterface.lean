import bench.tritonbench_g.rotary_emb_nopad.Proofs.CachedIOBindings

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem c_value (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (cInputs g)) (s : State α)
    (h : (cachedIO rest ops g p q).InputBinding v s)
    (key odd : Bool) (i : CLane g) (live : cMask g p key i) :
    Typed.cValue ops (Typed.cLayout g.request) ((cInvocation rest ops g p q).initial s) key odd i=
      Cell.mk .real (rotation ops (cPayload g v key Bool.false i) (cPayload g v key Bool.true i)
        (cCoeff g v Bool.false i) (cCoeff g v Bool.true i) odd) := by
  unfold Typed.cValue
  simp only [c_input_offset g p q _ (c_pid0 rest ops g p q s) (c_pid1 rest ops g p q s),
    c_coefficient g q _ (c_pid1 rest ops g p q s),Invocation.initial_mem]
  change Cell.mk .real (rotation ops
    ((s.mem (region key) (cInput g p q key Bool.false i)).read .real)
    ((s.mem (region key) (cInput g p q key Bool.true i)).read .real)
    ((s.mem (coefficient Bool.false) (cCoefficient g q Bool.false i)).read .real)
    ((s.mem (coefficient Bool.true) (cCoefficient g q Bool.true i)).read .real) odd)=_
  rw [c_input rest ops g p q v s h key Bool.false i live,
    c_input rest ops g p q v s h key Bool.true i live,
    c_coeff_input rest ops g p q v s h Bool.false i live.1,
    c_coeff_input rest ops g p q v s h Bool.true i live.1]

theorem c_frame (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (cInputs g)) (s : State α)
    (h : (cachedIO rest ops g p q).InputBinding v s) :
    (cachedIO rest ops g p q).Frame v s
      (Typed.cReference ops (Typed.cLayout g.request) ((cInvocation rest ops g p q).initial s)) := by
  let t := (cInvocation rest ops g p q).initial s
  have hp : t.pids 0=p := c_pid0 rest ops g p q s
  have hq : t.pids 1=q := c_pid1 rest ops g p q s
  intro r o outside
  by_cases head : p<g.qHeads
  · have md := c_metadata rest ops g p q v s h head
    have away (key : Bool) :
        Pair.Outside (Typed.cRegion (Typed.cLayout g.request) key)
          (Typed.cActive (Typed.cLayout g.request) t key) (Typed.cOffset (Typed.cLayout g.request) t key) r o := by
      by_cases hr : r=Typed.cRegion (Typed.cLayout g.request) key
      · right
        intro odd i active
        have live : cMask g p key i := (c_active g p t hp key i).mp ⟨head,active⟩
        rw [c_output_offset g p q t hp hq v md.1 md.2]
        cases key <;> cases odd
        all_goals first
          | simpa [cachedIO,cMemory,cOutputs,hr,Typed.cRegion,Typed.cLayout] using outside.1 (0 : Fin 4) i live
          | simpa [cachedIO,cMemory,cOutputs,hr,Typed.cRegion,Typed.cLayout] using outside.1 (1 : Fin 4) i live
          | simpa [cachedIO,cMemory,cOutputs,hr,Typed.cRegion,Typed.cLayout] using outside.1 (2 : Fin 4) i live
          | simpa [cachedIO,cMemory,cOutputs,hr,Typed.cRegion,Typed.cLayout] using outside.1 (3 : Fin 4) i live
      · exact Or.inl hr
    change (Typed.cReference ops (Typed.cLayout g.request) t).mem r o=s.mem r o
    have ht : t.pids 0<(Typed.cLayout g.request).Q_HEAD_NUM := by simpa [hp,Typed.cLayout,Geometry.request] using head
    unfold Typed.cReference
    rw [if_pos ht]
    unfold Typed.cBranch
    rw [Pair.frame _ _ _ _ _ _ _ (away Bool.false),Pair.frame _ _ _ _ _ _ _ (away Bool.true)]
    exact Invocation.initial_mem _ _ |> fun hmem => congrFun (congrFun hmem r) o
  · change (Typed.cReference ops (Typed.cLayout g.request) t).mem r o=s.mem r o
    have ht : ¬t.pids 0<(Typed.cLayout g.request).Q_HEAD_NUM := by simpa [hp,Typed.cLayout,Geometry.request] using head
    simp only [Typed.cReference,ht,↓reduceIte]
    exact Invocation.initial_mem _ _ |> fun hmem => congrFun (congrFun hmem r) o

theorem cached_implements (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (token : q<g.tokens) :
    (cachedIO rest ops g p q).ImplementsOn (MetadataDomain g) (cTarget ops g) := by
  intro v domain bounds hb s h
  refine ⟨(scheduleSafe_singleton _ _ _).mpr (c_safe rest ops g p q token v bounds hb s h),?_⟩
  let t := (cInvocation rest ops g p q).initial s
  have hp : t.pids 0=p := c_pid0 rest ops g p q s
  have hq : t.pids 1=q := c_pid1 rest ops g p q s
  have ht : t.pids 1<g.request.tokens := by simpa [hq,Geometry.request] using token
  let final := (Typed.cachedExecution rest ops (Typed.cLayout g.request) t).val
  have mem : final.mem=(Typed.cReference ops (Typed.cLayout g.request) t).mem :=
    Typed.cached_memory rest ops _ _ ht
  refine ⟨final,?_,?_,?_⟩
  · exact (runSchedule_singleton _ _).trans (Typed.cachedExecution rest ops (Typed.cLayout g.request) t).property
  · intro o i live
    have value (key odd : Bool) (i : CLane g) (hi : cMask g p key i) :
        final.mem (if key then "Cache" else "Q")
          (if key then cacheOffset g p v odd i else cInput g p q Bool.false odd i)=
            .mk .real (rotation ops (cPayload g v key Bool.false i) (cPayload g v key Bool.true i)
              (cCoeff g v Bool.false i) (cCoeff g v Bool.true i) odd) := by
      have active := (c_active g p t hp key i).mpr hi
      have md := c_metadata rest ops g p q v s h active.1
      have past' : 0≤Typed.cPast (Typed.cLayout g.request) t := by rw [md.1]; exact domain.1
      have page' : 0≤Typed.cBlock (Typed.cLayout g.request) t := by rw [md.2]; exact domain.2
      have head' : t.pids 0<g.request.qHeads := by simpa [hp,Geometry.request] using active.1
      have result := Typed.c_readback ops g.request t past' page' head' key odd i active.2
      rw [c_output_offset g p q t hp hq v md.1 md.2,
        c_value rest ops g p q v s h key odd i hi] at result
      rw [mem]
      exact result
    change Fin 4 at o
    fin_cases o
    · change mapCell id (final.mem "Q" (cInput g p q Bool.false Bool.false i)) = _
      have hv := value Bool.false Bool.false i live
      simp only [Bool.false_eq_true,↓reduceIte] at hv
      rw [hv]
      rfl
    · change mapCell id (final.mem "Q" (cInput g p q Bool.false Bool.true i)) = _
      have hv := value Bool.false Bool.true i live
      simp only [Bool.false_eq_true,↓reduceIte] at hv
      rw [hv]
      rfl
    · change mapCell id (final.mem "Cache" (cacheOffset g p v Bool.false i)) = _
      have hv := value Bool.true Bool.false i live
      simp only [Bool.false_eq_true,↓reduceIte] at hv
      rw [hv]
      rfl
    · change mapCell id (final.mem "Cache" (cacheOffset g p v Bool.true i)) = _
      have hv := value Bool.true Bool.true i live
      simp only [Bool.false_eq_true,↓reduceIte] at hv
      rw [hv]
      rfl
  · intro r o outside
    rw [mem]
    exact c_frame rest ops g p q v s h r o outside

#axiomsClean cached_implements
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
