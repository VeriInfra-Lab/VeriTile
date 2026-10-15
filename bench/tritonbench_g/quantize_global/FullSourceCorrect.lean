import bench.tritonbench_g.quantize_global.Proofs.FinitePhysicalExecution

namespace QuantizeGlobalFinite
set_option maxRecDepth 32000
set_option maxHeartbeats 1000000

@[simp] theorem except_bind_ok {ε α β : Type} (x : α) (f : α → Except ε β) :
    (Except.ok x >>= f)=f x := rfl
@[simp] theorem except_bind_error {ε α β : Type} (e : ε) (f : α → Except ε β) :
    (Except.error e >>= f)=Except.error e := rfl

structure Admitted (env : Environment) (device : Device) (d : DType) (n : Nat)
    (rankZero : Bool) (r : Storage) : Prop where
  signature : route d=some r
  cuda : device=.cuda
  nonempty : 0<n
  rank : rankZero=false
  abs : env .abs=none
  maximum : env .maximum=none
  unsqueeze : env .unsqueeze=none
  reciprocal : env .reciprocal=none
  output : env .outputAllocation=none
  signaturePrimitive : env .signature=none
  compilation : env .compilation=none
  launch : env .launch=none

theorem admission_binding (env : Environment) (device : Device) (d : DType) (n : Nat)
    (rankZero : Bool) (r : Storage)
    (h : admission env device d n rankZero=.ok r) : Admitted env device d n rankZero r := by
  unfold admission at h
  simp only [Bind.bind,Except.bind,Pure.pure,Except.pure] at h
  have ha : env .abs=none := by
    cases he : env .abs <;> simp_all only [reduceCtorEq]
  simp only [ha] at h
  split at h <;> try { cases h }
  have hm : env .maximum=none := by
    cases he : env .maximum <;> simp_all only [reduceCtorEq]
  simp only [hm] at h
  split at h <;> try { cases h }
  split at h <;> try { cases h }
  have hu : env .unsqueeze=none := by
    cases he : env .unsqueeze <;> simp_all only [reduceCtorEq]
  simp only [hu] at h
  have hr : env .reciprocal=none := by
    cases he : env .reciprocal <;> simp_all only [reduceCtorEq]
  simp only [hr] at h
  split at h <;> try { cases h }
  have ho : env .outputAllocation=none := by
    cases he : env .outputAllocation <;> simp_all only [reduceCtorEq]
  simp only [ho] at h
  split at h <;> try { cases h }
  have hs : env .signature=none := by
    cases he : env .signature <;> simp_all only [reduceCtorEq]
  simp only [hs] at h
  split at h <;> try { cases h }
  have hc : env .compilation=none := by
    cases he : env .compilation <;> simp_all only [reduceCtorEq]
  simp only [hc] at h
  split at h <;> try { cases h }
  have hl : env .launch=none := by
    cases he : env .launch <;> simp_all only [reduceCtorEq]
  simp only [hl] at h
  cases ht : route d with
  | none => simp [ht] at h
  | some r' =>
    simp only [ht,Except.ok.injEq] at h
    subst r'
    constructor
    · exact ht
    · cases device <;> simp_all
    · omega
    · cases rankZero <;> simp_all
    all_goals assumption


theorem execution_environment (b : Numerics) (env : Environment) (device : Device)
    (d : DType) (r : Storage) (a : Input) (alloc : Allocations) (m : Memory)
    (x : String) (trials : List Config) (last : Config)
    (A : Admitted env device d a.numel (a.shape==[]) r) :
    execute b env r a alloc m x trials last=execute b (fun _ => none) r a alloc m x trials last := by
  have hp : prepare b env r a alloc m x=prepare b (fun _ => none) r a alloc m x := by
    simp only [prepare,primitive,A.abs,A.maximum,A.unsqueeze,A.reciprocal,A.output]
  simp only [execute,hp,A.signaturePrimitive,A.compilation,A.launch]

/-- Ordinary data/metadata/allocation readiness is required only when the
source actually reaches a bounded, in-storage typed kernel route. This never
asserts admission success, defined arithmetic, a successful run or an output. -/
def Ready (env : Environment) (d : DType) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (p : PhysicalLayout) : Prop :=
  ∀ r, admission env (if a.isCuda then .cuda else .cpu) d a.numel (a.shape==[])=.ok r →
    a.numel≤2^31 → a.storageOffset+a.numel≤a.storageLength →
    ∃ logical xs, Domain r a alloc m x logical xs ∧ PhysicalBinding r a alloc m x p

/-- Complete finite original API outcome contract. Every dtype and explicit
primitive failure remains in the source-ordered admission relation. No Safe or
successful-execution premise is used: all finite words either reach an undefined
active conversion or actual returned values. Successful results include an
actual primitive byte-store trace through every autotuning trial, concrete
physical results/frames and exact return metadata. Hazard labels classify the
source; they are not new native checks or claims of a native trap/rollback. -/
specification quantize_global_full_source_outcomes
    (b : Numerics) (env : Environment) (d : DType) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (p : PhysicalLayout) (size : a.numel<2^63)
    (ready : Ready env d a alloc m x p) (trials : List Config) (last : Config) :
    (∃ why, admission env (if a.isCuda then .cuda else .cpu) d a.numel (a.shape==[])=.error why ∧
      executeAPI b env d a alloc m x trials last=.rejected why) ∨
    (∃ r, admission env (if a.isCuda then .cuda else .cpu) d a.numel (a.shape==[])=.ok r ∧
      2^31<a.numel ∧ executeAPI b env d a alloc m x trials last=.hazardous .i32OffsetWrap) ∨
    (∃ r, admission env (if a.isCuda then .cuda else .cpu) d a.numel (a.shape==[])=.ok r ∧
      a.numel≤2^31 ∧ UnsafeView a ∧
      executeAPI b env d a alloc m x trials last=.hazardous .linearViewOverrun) ∨
    (∃ (r : Storage) (logical : List (Word r)) (xs : Nat → Word r) (maximum : Word r),
      route d=some r ∧ Domain r a alloc m x logical xs ∧ PhysicalBinding r a alloc m x p ∧
      produced b r logical=some maximum ∧
      ((∃ i, executeAPI b env d a alloc m x trials last=
        .attempted r (.error (.interpreter (.conversionUndefined i)))) ∨
       (∃ ret, executeAPI b env d a alloc m x trials last=.attempted r (.ok ret) ∧
        Result b r a alloc m xs maximum ret ∧
        PhysicalResult b r a alloc p xs maximum ret (externalBytes r a alloc p ret) ∧
        PhysicalStores a alloc p (allocated r a alloc m maximum (b.reciprocal r maximum))
          (hostPhysical r p maximum (b.reciprocal r maximum))
          ret.memory (externalBytes r a alloc p ret)))) := by
  cases ha : admission env (if a.isCuda then .cuda else .cpu) d a.numel (a.shape==[]) with
  | error why =>
    left
    exact ⟨why,rfl,quantize_global_rejected_source_outcome b env d a alloc m x trials last why ha⟩
  | ok r =>
    have A := admission_binding env (if a.isCuda then .cuda else .cpu) d a.numel (a.shape==[]) r ha
    by_cases large : 2^31<a.numel
    · right; left
      exact ⟨r,rfl,large,quantize_global_unsafe_source_outcome b env d r a alloc m x trials last ha large⟩
    · have bounded : a.numel≤2^31 := by omega
      by_cases overrun : UnsafeView a
      · right; right; left
        exact ⟨r,rfl,bounded,overrun,
          quantize_global_unsafe_view_outcome b env d r a alloc m x trials last ha bounded overrun⟩
      · have span : a.storageOffset+a.numel≤a.storageLength := by unfold UnsafeView at overrun; omega
        obtain ⟨logical,xs,D,P⟩ := ready r ha bounded span
        have cuda : a.isCuda=true := by
          have h:=A.cuda
          cases hz : a.isCuda <;> simp_all
        have rank : a.shape≠[] := by
          intro h
          have hz:=A.rank
          simp [h] at hz
        have hz : (a.shape==[])=false := by simpa using A.rank
        have hno : admission (fun _ => none) (if a.isCuda then .cuda else .cpu)
            (dtype r) a.numel (a.shape==[])=.ok r := by
          simp only [cuda,if_true,hz]
          exact admitted_route r a.numel A.nonempty
        have henv := execution_environment b env _ d r a alloc m x trials last A
        have eno : executeAPI b (fun _ => none) (dtype r) a alloc m x trials last=
            .attempted r (execute b (fun _ => none) r a alloc m x trials last) := by
          simp only [executeAPI,hno,if_neg large,if_neg (Nat.not_lt_of_ge span)]
        have same : executeAPI b env d a alloc m x trials last=
            executeAPI b (fun _ => none) (dtype r) a alloc m x trials last := by
          rw [eno]
          simp only [executeAPI,ha,if_neg large,if_neg (Nat.not_lt_of_ge span),henv]
        obtain ⟨maximum,hm,outcome⟩ := quantize_global_all_finite_value_outcomes b r a alloc m x
          logical xs D A.nonempty rank cuda trials last
        right; right; right
        refine ⟨r,logical,xs,maximum,A.signature,D,P,hm,?_⟩
        rcases outcome with bad|good
        · left; obtain ⟨i,hi⟩:=bad; exact ⟨i,by rw [same]; exact hi⟩
        · right
          obtain ⟨ret,hr,R⟩:=good
          have hexec : execute b (fun _ => none) r a alloc m x trials last=.ok ret := by
            rw [eno] at hr
            simpa only [APIResult.attempted.injEq,true_and] using hr
          have hp := prepare_success b r a alloc m x logical maximum D.logical hm rank cuda
          have trace := execute_physical b r a alloc m x p D.fresh _ hp trials last ret hexec
          dsimp only at trace
          rw [host_physical_preparation r a alloc m x p D.fresh P maximum (b.reciprocal r maximum)] at trace
          exact ⟨ret,by rw [same]; exact hr,R,
            quantize_global_external_result b r a alloc m x p xs maximum ret P R,trace⟩

#specNonCircular Ready avoiding [executeAPI,execute,prepare,runTrials,runGrid,runProgram,runLanes,source,Expr.eval]
#axiomsClean admission_binding
#axiomsClean quantize_global_full_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms

end QuantizeGlobalFinite
