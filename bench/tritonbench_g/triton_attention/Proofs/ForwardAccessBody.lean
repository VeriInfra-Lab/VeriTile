import bench.tritonbench_g.triton_attention.Proofs.AccessGeometry
import bench.tritonbench_g.triton_attention.Proofs.FPForwardBody

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem forward_score_safe (rest : Algebra α) (P : Primitives α)
    (p : ForwardParameters) (initial current : State α) (block : Nat)
    (bounds : RegionName→Nat)
    (pointer : current.regs .blockPtr [p.blockN,p.head] "k_tile_ptr"=
      some (fun _=>forwardKVPointer p initial p.k block))
    (keys : ∀ i : FKey p,forwardKVOffset p initial block i<bounds p.k) :
    Safety.traceSafeList (backend rest P) bounds ((forwardBody p).take 5) current := by
  change Safety.traceSafeList _ _ [_ , _ , _ , _ , _] _
  apply Safety.traceSafeList_cons_intro
  · simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,Option.bind,pointer,forwardKVPointer,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    intro i d _
    exact keys (i,d,PUnit.unit)
  · intro next _
    apply AccessSafety.list_safe_of_steps
    intro statement member state
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem forward_middle_safe (M : Algebra α) (p : ForwardParameters)
    (s : State α) (bounds : RegionName→Nat) :
    Safety.traceSafeList M bounds (((forwardBody p).drop 5).take 8) s := by
  apply AccessSafety.list_safe_of_steps
  intro statement member state
  change statement∈[_,_,_,_,_,_,_,_] at member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem forward_accumulate_safe (rest : Algebra α) (P : Primitives α)
    (p : ForwardParameters) (initial current : State α) (block : Nat)
    (bounds : RegionName→Nat)
    (pointer : current.regs .blockPtr [p.blockN,p.head] "v_tile_ptr"=
      some (fun _=>forwardKVPointer p initial p.v block))
    (values : ∀ i : FKey p,forwardKVOffset p initial block i<bounds p.v) :
    Safety.traceSafeList (backend rest P) bounds ((forwardBody p).drop 13) current := by
  change Safety.traceSafeList _ _ [_ , _ , _ , _ , _ , _] _
  apply Safety.traceSafeList_cons_intro
  · simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,Option.bind,pointer,forwardKVPointer,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    intro i d _
    exact values (i,d,PUnit.unit)
  · intro next _
    apply AccessSafety.list_safe_of_steps
    intro statement member state
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem forward_body_safe (rest : Algebra α) (P : Primitives α)
    (p : ForwardParameters) (initial current : State α) (block : Nat)
    (acc : ForwardAccumulator α p) (hn : 0<p.blockN)
    (inv : ForwardInvariant P p initial block acc current) (bounds : RegionName→Nat)
    (keys : ∀ i : FKey p,forwardKVOffset p initial block i<bounds p.k)
    (values : ∀ i : FKey p,forwardKVOffset p initial block i<bounds p.v) :
    Safety.traceSafeList (backend rest P) bounds (forwardBody p)
      (current.setReg "start_n" .nat [] (fun _=>block*p.blockN)) := by
  rw [← List.take_append_drop 5 (forwardBody p),Safety.traceSafeList_append]
  refine ⟨forward_score_safe rest P p initial _ block bounds (by simpa using inv.keyPtr) keys,?_⟩
  intro scored ran
  rw [forwardScoreExecution rest P p initial current block acc inv] at ran
  cases ran
  let score := forwardScoreResult P p initial current block
  have scoreInv := forwardScoreInvariant P p initial current block acc inv
  have split : (forwardBody p).drop 5=
      ((forwardBody p).drop 5).take 5++((forwardBody p).drop 10).take 3++(forwardBody p).drop 13 := rfl
  rw [split,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨?_,?_⟩,?_⟩
  · apply AccessSafety.list_safe_of_steps
    intro statement member state
    change statement∈[_,_,_,_,_] at member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  · intro next _
    apply AccessSafety.list_safe_of_steps
    intro statement member state
    change statement∈[_,_,_] at member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  · intro next ran
    rw [run_append,forwardSoftmaxExecution rest P p initial score block acc hn
      scoreInv.maximum scoreInv.denominator (by simp [score,forwardScoreResult])] at ran
    simp only [Option.bind_some] at ran
    rw [forwardNormalizeExecution rest P p initial _ block acc
      (by simp [forwardSoftmaxResult]) (by simp [forwardSoftmaxResult])
      (by simp [forwardSoftmaxResult]) (by simp [forwardSoftmaxResult,score,forwardScoreResult,inv.numerator])] at ran
    cases ran
    apply forward_accumulate_safe rest P p initial _ block bounds _ values
    simp [forwardNormalizeResult,forwardSoftmaxResult,score,forwardScoreResult,inv.valuePtr]

#axiomsClean forward_body_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
