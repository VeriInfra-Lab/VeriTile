import bench.tritonbench_g.rotary_emb_nopad.Kernels
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.AddressCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Proof-only decomposition; neither authoritative source is generated here. -/
def uPrefix (p : Kernels.UncachedLayout) := (Kernels.uncached p).surfaceBody.take 11
def uSharedPrefix (p : Kernels.UncachedLayout) := (Kernels.uncachedShared p).surfaceBody.take 11
def uSuffix (p : Kernels.UncachedLayout) := (Kernels.uncached p).surfaceBody.drop 11

theorem u_body (p : Kernels.UncachedLayout) :
    (Kernels.uncachedShared p).surfaceBody = uSharedPrefix p ++ uSuffix p := rfl

theorem u_prefix_run (M : Algebra α) (p : Kernels.UncachedLayout) (s : State α) :
    run M (uPrefix p) s = run M (uSharedPrefix p) s := by
  simp [uPrefix,uSharedPrefix,Kernels.uncached,Kernels.uncachedShared,
    ComputeKernel.surfaceBody,List.take,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLt,natLt,bop,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Option.bind,ComputeDType.eraseDType,
    Int.add_mul,Int.add_assoc]
  congr 1
  funext i
  dsimp [bop,TileShape.insertAxis,TileShape.dropInsertedIndex]
  ring

theorem u_prefix_safe (M : Algebra α) (p : Kernels.UncachedLayout)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (uPrefix p) s ↔
      Safety.traceSafeList M bounds (uSharedPrefix p) s := by
  simp (config := { maxSteps := 200000 }) [uPrefix,uSharedPrefix,
    Kernels.uncached,Kernels.uncachedShared,ComputeKernel.surfaceBody,List.take,
    Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,
    integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,Option.bind,
    ComputeDType.eraseDType,Int.add_mul,Int.add_assoc]
  try rfl

theorem uncached_execution (M : Algebra α) (p : Kernels.UncachedLayout) (s : State α) :
    exec M (Kernels.uncached p) s = exec M (Kernels.uncachedShared p) s := by
  change run M (Kernels.uncached p).surfaceBody s = run M (Kernels.uncachedShared p).surfaceBody s
  rw [u_body]
  conv_lhs => rw [← List.take_append_drop 11 (Kernels.uncached p).surfaceBody]
  change run M (uPrefix p ++ uSuffix p) s = _
  rw [run_append,run_append,u_prefix_run]

theorem uncached_safe (M : Algebra α) (p : Kernels.UncachedLayout)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (Kernels.uncached p) s ↔ TraceSafe M bounds (Kernels.uncachedShared p) s := by
  unfold TraceSafe
  rw [u_body]
  conv_lhs => rw [← List.take_append_drop 11 (Kernels.uncached p).surfaceBody]
  change Safety.traceSafeList M bounds (uPrefix p ++ uSuffix p) s ↔ _
  simp only [Safety.traceSafeList_append,u_prefix_run,u_prefix_safe]

def cBody (p : Kernels.CachedLayout) : List ComputeStmt :=
  match (Kernels.cached p).surfaceBody.drop 1 with
  | [.ifThen _ body] => body
  | _ => []
def cSharedBody (p : Kernels.CachedLayout) : List ComputeStmt :=
  match (Kernels.cachedShared p).surfaceBody.drop 1 with
  | [.ifThen _ body] => body
  | _ => []
def cCond (p : Kernels.CachedLayout) : Op .bool [] :=
  match (Kernels.cached p).surfaceBody.drop 1 with
  | [.ifThen cond _] => cond
  | _ => .constBool Bool.false

def cPrefix (p : Kernels.CachedLayout) := (cBody p).take 5
def cSharedPrefix (p : Kernels.CachedLayout) := (cSharedBody p).take 5
def cSuffix (p : Kernels.CachedLayout) := (cBody p).drop 5

theorem c_shared_body (p : Kernels.CachedLayout) :
    cSharedBody p = cSharedPrefix p ++ cSuffix p := rfl

theorem c_prefix_run (M : Algebra α) (p : Kernels.CachedLayout) (s : State α) :
    run M (cPrefix p) s = run M (cSharedPrefix p) s := by
  cases h : s.regs .int [] "block_head_index" with
  | none =>
    simp [cPrefix,cSharedPrefix,cBody,cSharedBody,Kernels.cached,Kernels.cachedShared,
      ComputeKernel.surfaceBody,List.drop,List.take,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,numeric,numericLt,natLt,bop,integralFloorDiv,integralMod,
      ofFloat,toFloat,Region.cast,Option.bind,ComputeDType.eraseDType,h]
  | some v =>
    simp [cPrefix,cSharedPrefix,cBody,cSharedBody,Kernels.cached,Kernels.cachedShared,
      ComputeKernel.surfaceBody,List.drop,List.take,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,numeric,numericLt,natLt,bop,integralFloorDiv,integralMod,
      ofFloat,toFloat,Region.cast,Option.bind,ComputeDType.eraseDType,h,
      Int.add_mul,Int.add_assoc]
    congr 1
    funext i
    dsimp [bop,TileShape.insertAxis,TileShape.dropInsertedIndex]
    ring

theorem c_prefix_safe (M : Algebra α) (p : Kernels.CachedLayout)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (cPrefix p) s ↔
      Safety.traceSafeList M bounds (cSharedPrefix p) s := by
  simp [cPrefix,cSharedPrefix,cBody,cSharedBody,Kernels.cached,Kernels.cachedShared,
    ComputeKernel.surfaceBody,List.drop,List.take,Safety.traceSafeList_cons,
    Safety.traceSafeList_nil,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem c_body_run (M : Algebra α) (p : Kernels.CachedLayout) (s : State α) :
    run M (cBody p) s = run M (cSharedBody p) s := by
  rw [c_shared_body]
  conv_lhs => rw [← List.take_append_drop 5 (cBody p)]
  change run M (cPrefix p ++ cSuffix p) s = _
  rw [run_append,run_append,c_prefix_run]

theorem c_body_safe (M : Algebra α) (p : Kernels.CachedLayout)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (cBody p) s ↔
      Safety.traceSafeList M bounds (cSharedBody p) s := by
  rw [c_shared_body]
  conv_lhs => rw [← List.take_append_drop 5 (cBody p)]
  change Safety.traceSafeList M bounds (cPrefix p ++ cSuffix p) s ↔ _
  simp only [Safety.traceSafeList_append,c_prefix_run,c_prefix_safe]

theorem c_tail_run (M : Algebra α) (p : Kernels.CachedLayout) (s : State α) :
    run M [.ifThen (cCond p) (cBody p)] s =
      run M [.ifThen (cCond p) (cSharedBody p)] s := by
  simp only [run,step,c_body_run]

theorem c_tail_safe (M : Algebra α) (p : Kernels.CachedLayout)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds [.ifThen (cCond p) (cBody p)] s ↔
      Safety.traceSafeList M bounds [.ifThen (cCond p) (cSharedBody p)] s := by
  simp only [Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,c_body_safe]
  simp

theorem cached_execution (M : Algebra α) (p : Kernels.CachedLayout) (s : State α) :
    exec M (Kernels.cached p) s = exec M (Kernels.cachedShared p) s := by
  have oldBody : (Kernels.cached p).surfaceBody =
      (Kernels.cached p).surfaceBody.take 1 ++ [.ifThen (cCond p) (cBody p)] := rfl
  have newBody : (Kernels.cachedShared p).surfaceBody =
      (Kernels.cached p).surfaceBody.take 1 ++ [.ifThen (cCond p) (cSharedBody p)] := rfl
  change run M (Kernels.cached p).surfaceBody s = run M (Kernels.cachedShared p).surfaceBody s
  rw [oldBody,newBody,run_append,run_append]
  exact congrArg (fun f => (run M ((Kernels.cached p).surfaceBody.take 1) s).bind f)
    (funext (c_tail_run M p))

theorem cached_safe (M : Algebra α) (p : Kernels.CachedLayout)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (Kernels.cached p) s ↔ TraceSafe M bounds (Kernels.cachedShared p) s := by
  have oldBody : (Kernels.cached p).surfaceBody =
      (Kernels.cached p).surfaceBody.take 1 ++ [.ifThen (cCond p) (cBody p)] := rfl
  have newBody : (Kernels.cachedShared p).surfaceBody =
      (Kernels.cached p).surfaceBody.take 1 ++ [.ifThen (cCond p) (cSharedBody p)] := rfl
  unfold TraceSafe
  rw [oldBody,newBody]
  simp only [Safety.traceSafeList_append]
  exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right
    (fun _ => c_tail_safe M p bounds t)))

#axiomsClean uncached_execution
#axiomsClean uncached_safe
#axiomsClean cached_execution
#axiomsClean cached_safe
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.AddressCSE
