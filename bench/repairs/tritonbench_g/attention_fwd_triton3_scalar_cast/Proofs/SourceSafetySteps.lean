import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Program
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetySteps

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
    traceSafeList (engine o.algebra) bounds ((parts c).before.take 16) s := by
  apply pure_safe
  intro stmt member current
  simp [parts,Kernels.source,Quoted.kernel,ComputeKernel.surfaceBody,Source.intRange?] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try rw [safeBlockPtr2]
  all_goals simp only [safeOp_unfold,true_and,and_true]

theorem initialization_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat)
    (initialState s : State α) (live : HeaderMatches c (inputs initialState) s)
    (backed : PreludeBounds c (inputs initialState) bounds) :
    traceSafeList (engine o.algebra) bounds (((parts c).before.drop 16).take 4) s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6⟩
  rcases backed with ⟨query,center,denominator,output⟩
  by_cases hi : c.init=0 <;> by_cases he : c.params.evenM=0
  all_goals
    simp (config := {maxSteps := 400000}) [parts,Kernels.source,Quoted.kernel,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,hi,he,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp only [block_bounds_2d,he,ne_eq,not_true_eq_false,not_false_eq_true,if_true,if_false,BlockPtr.inBounds_nil_checkedAxes,forall_true_left] at query output
  all_goals
    first
    | exact ⟨⟨center,denominator (Or.inl hi),output⟩,query⟩
    | exact query

theorem body_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (cur : Int)
    (key : BlockBounds bounds old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1]))
    (value : BlockBounds bounds old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (engine o.algebra) bounds (parts c).body
      (s.setReg "start_n" .int [] (fun _ => cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12⟩
  by_cases he : c.params.evenN=0 <;> by_cases hw : c.window=0 <;> by_cases hc : c.complement=0
  all_goals
    simp (config := {maxSteps := 1000000}) [parts,Kernels.source,Quoted.kernel,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,he,hw,hc,positive,TileShape.axisDim,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp only [block_bounds_2d,he,ne_eq,not_true_eq_false,not_false_eq_true,if_true,if_false,BlockPtr.inBounds_nil_checkedAxes,forall_true_left] at key value
  all_goals exact ⟨key,value⟩

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
