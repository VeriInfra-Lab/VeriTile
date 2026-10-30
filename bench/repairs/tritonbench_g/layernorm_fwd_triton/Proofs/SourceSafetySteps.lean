import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceComplete
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelSafety
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

private theorem list_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ st ∈ code,∀ s : State α,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun e he => safe e (by simp [he])) t

theorem prelude_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (prelude p) s := by
  apply list_safe
  simp only [prelude,Source.kernel,runtimeSource,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    List.take_succ_cons,List.take_zero]
  intro st member t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]

theorem mean_post_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (meanPost p) s := by
  apply list_safe
  simp only [meanPost,Source.kernel,runtimeSource,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  intro st member t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]

theorem var_post_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (varPost p) s := by
  apply list_safe
  simp only [varPost,Source.kernel,runtimeSource,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  intro st member t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]

theorem mean_body_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (off : Nat)
    (a : TileIndex [p.block] → α) (ready : MeanReady M p s t a)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x) :
    Safety.traceSafeList M bounds (loopBody p 18) (t.setReg "off" .int [] (fun _ => (off : Int))) := by
  simp [loopBody,Source.kernel,runtimeSource,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add,Int.toNat_add,Int.toNat_mul,Int.natCast_nonneg,Int.toNat_of_nonpos,ready.x, ready.nValue, ready.zeroValue, ready.stepValue, ready.lanes, ready.xStride, ready.yStride, ready.wStride,ready.accumulator]
  exact fun j active => input (off+j.val) active

theorem var_body_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (off : Nat)
    (a : TileIndex [p.block] → α) (ready : VarReady M p s t a)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x) :
    Safety.traceSafeList M bounds (loopBody p 21) (t.setReg "off" .int [] (fun _ => (off : Int))) := by
  simp [loopBody,Source.kernel,runtimeSource,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add,Int.toNat_add,Int.toNat_mul,Int.natCast_nonneg,Int.toNat_of_nonpos,ready.x, ready.nValue, ready.zeroValue, ready.stepValue, ready.lanes, ready.xStride, ready.yStride, ready.wStride,ready.accumulator,ready.mean]
  exact fun j active => input (off+j.val) active

theorem output_body_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (off : Nat)
    (mem : RegionName → Nat → Cell α) (ready : OutputReady M p s t mem)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x)
    (weight : ∀ j,j<p.columns → wBase p s+j*p.weightDimStride<bounds p.weight)
    (output : ∀ j,j<p.columns → yBase p s+j*p.yDimStride<bounds p.y) :
    Safety.traceSafeList M bounds (loopBody p 24) (t.setReg "off" .int [] (fun _ => (off : Int))) := by
  simp [loopBody,Source.kernel,runtimeSource,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add,Int.toNat_add,Int.toNat_mul,Int.natCast_nonneg,Int.toNat_of_nonpos,ready.x, ready.nValue, ready.zeroValue, ready.stepValue, ready.lanes, ready.xStride, ready.yStride, ready.wStride,ready.y,ready.w,ready.mean,ready.rstd]
  exact ⟨fun j active => weight (off+j.val) active,
    fun j active => input (off+j.val) active,fun j active => output (off+j.val) active⟩

#axiomsClean prelude_safe
#axiomsClean mean_post_safe
#axiomsClean var_post_safe
#axiomsClean mean_body_safe
#axiomsClean var_body_safe
#axiomsClean output_body_safe
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelSafety
