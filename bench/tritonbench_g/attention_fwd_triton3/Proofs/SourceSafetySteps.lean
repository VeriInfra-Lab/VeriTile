import bench.tritonbench_g.attention_fwd_triton3.Proofs.MemoryFree
import bench.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetySteps

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Safety FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
open Reference
open VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety (safeBlockPtr2 pure_safe)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def BlockBounds (bounds : RegionName → Nat) (ptr : BlockPtr) (shape : TileShape)
    (checked : List Nat) : Prop := ∀ i : TileIndex shape,
  ptr.inBounds (TileShape.indexToList shape i) checked=Bool.true →
    ptr.address (TileShape.indexToList shape i)<bounds ptr.region


theorem block_bounds_2d (bounds : RegionName → Nat) (ptr : BlockPtr) (m n : Nat)
    (checked : List Nat) : BlockBounds bounds ptr [m,n] checked ↔
      ∀ (r : Fin m) (d : Fin n),
        ptr.inBounds (TileShape.indexToList [m,n] (r,d,PUnit.unit)) checked=Bool.true →
        ptr.address (TileShape.indexToList [m,n] (r,d,PUnit.unit))<bounds ptr.region := by
  constructor
  · intro h r d
    exact h (r,d,PUnit.unit)
  · intro h ⟨r,d,tail⟩
    cases tail
    exact h r d

structure PreludeBounds (c : Call) (a : Inputs α) (bounds : RegionName → Nat) : Prop where
  query : BlockBounds bounds (qptr c a) [c.BM,c.D] (if c.params.evenM≠0 then [] else [0,1])
  center : ∀ i : Fin c.BM,carryAddress c a i.val<bounds "M"
  denominator : c.init=0 ∨ c.finish=0 → ∀ i : Fin c.BM,carryAddress c a i.val<bounds "L"
  output : BlockBounds bounds (optr c a) [c.BM,c.D] []

theorem header_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine o.algebra) bounds ((parts c).before.take 17) s := by
  apply pure_safe
  intro stmt member current
  simp [parts,Kernels.source,Quoted.kernel,DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,sourceKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,ComputeKernel.surfaceBody,Source.intRange?] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try rw [safeBlockPtr2]
  all_goals simp only [safeOp_unfold,true_and,and_true]

theorem initialization_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat)
    (initialState s : State α) (live : HeaderMatches c o (inputs initialState) s)
    (backed : PreludeBounds c (inputs initialState) bounds) :
    traceSafeList (engine o.algebra) bounds (((parts c).before.drop 17).take 4) s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,hscale⟩
  rcases backed with ⟨query,center,denominator,output⟩
  by_cases hi : c.init=0 <;> by_cases he : c.params.evenM=0
  all_goals
    simp (config := {maxSteps := 400000}) [parts,Kernels.source,Quoted.kernel,
      DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,hscale,hi,he,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp only [block_bounds_2d,he,ne_eq,not_true_eq_false,not_false_eq_true,if_true,if_false,BlockPtr.inBounds_nil_checkedAxes,forall_true_left] at query output
  all_goals
    first
    | exact ⟨⟨center,denominator (Or.inl hi),output⟩,query⟩
    | exact query

theorem score_load_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (cur : Int)
    (key : BlockBounds bounds old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (engine o.algebra) bounds ((parts c).body.take 2)
      (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _=>cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16⟩
  by_cases he : c.params.evenN=0
  all_goals
    simp (config := {maxSteps := 1000000}) [parts,Kernels.source,Quoted.kernel,
      DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,he,positive,TileShape.axisDim,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simpa only [block_bounds_2d,he,ne_eq,not_true_eq_false,not_false_eq_true,if_true,if_false,
      BlockPtr.inBounds_nil_checkedAxes,forall_true_left] using key

theorem score_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (cur : Int)
    (key : BlockBounds bounds old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (engine o.algebra) bounds ((parts c).body.take 7)
      (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _=>cur)) := by
  have split : (parts c).body.take 7=(parts c).body.take 2 ++ ((parts c).body.drop 2).take 5 := rfl
  rw [split,traceSafeList_append]
  refine ⟨score_load_safe c o positive bounds initialState s old live cur key,?_⟩
  intro t _
  prove_memory_free

theorem update_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine o.algebra) bounds (((parts c).body.drop 7).take 12) s := by
  prove_memory_free

theorem product_load_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (value : BlockBounds bounds old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (engine o.algebra) bounds (((parts c).body.drop 19).take 1) s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16⟩
  by_cases he : c.params.evenN=0
  all_goals
    simp (config := {maxSteps := 1000000}) [parts,Kernels.source,Quoted.kernel,
      DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,he,positive,TileShape.axisDim,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simpa only [block_bounds_2d,he,ne_eq,not_true_eq_false,not_false_eq_true,if_true,if_false,
      BlockPtr.inBounds_nil_checkedAxes,forall_true_left] using value

theorem product_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (value : BlockBounds bounds old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (engine o.algebra) bounds ((parts c).body.drop 19) s := by
  have split : (parts c).body.drop 19=((parts c).body.drop 19).take 1 ++ (parts c).body.drop 20 := rfl
  rw [split,traceSafeList_append]
  refine ⟨product_load_safe c o positive bounds initialState s old live value,?_⟩
  intro t _
  prove_memory_free

theorem body_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (cur : Int) (memory : s.mem=initialState.mem)
    (key : BlockBounds bounds old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1]))
    (value : BlockBounds bounds old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (engine o.algebra) bounds (parts c).body
      (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _ => cur)) := by
  obtain ⟨scored,hscore,scoreReady,scoreMemory⟩:=Reference.score_phase c o positive initialState s old live memory cur
  obtain ⟨updated,hupdate,updateReady,updateMemory⟩:=
    Reference.update_phase c o positive initialState scored old cur scoreReady scoreMemory
  have split : (parts c).body=(parts c).body.take 7 ++
      (((parts c).body.drop 7).take 12 ++ (parts c).body.drop 19) := rfl
  rw [split]
  exact append_of_run (score_safe c o positive bounds initialState s old live cur key) hscore
    (append_of_run (update_safe c o bounds scored) hupdate
      (product_safe c o positive bounds initialState updated (scaledCarry c o (inputs initialState) cur old)
        updateReady.1 value))

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
