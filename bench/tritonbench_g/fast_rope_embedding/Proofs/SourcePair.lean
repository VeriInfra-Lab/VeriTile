import bench.tritonbench_g.fast_rope_embedding.Kernels
import bench.tritonbench_g.fast_rope_embedding.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.SourcePair
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
set_option maxHeartbeats 5000000
noncomputable section

namespace Old
abbrev source := VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource.rope
end Old

theorem original_source (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Kernels.rope Q C S QR CR SR seq H heads B backwards =
      Old.source .fp32 .fp32 .fp32 Bool.false Q C S QR CR SR seq H heads B backwards := by
  cases backwards <;> rfl

theorem candidate_source (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Kernels.ropeIndexCommuted Q C S QR CR SR seq H heads B backwards =
      Old.source .fp32 .fp32 .fp32 Bool.true Q C S QR CR SR seq H heads B backwards := by
  cases backwards <;> rfl

/-- Proof-only decomposition of the one changed integer assignment. -/
def headStep (candidate : Bool) : ComputeStmt :=
  let a : Op .nat [] := .ref .nat [] "group_head_position"
  let b : Op .nat [] := .ref .nat [] "ROPE_GROUP_SIZE"
  .assign .nat [] "head_start" (.alg (if candidate then .mul .nat .nil b a else .mul .nat .nil a b))

def source (candidate : Bool) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :=
  if candidate then Kernels.ropeIndexCommuted Q C S QR CR SR seq H heads B backwards
  else Kernels.rope Q C S QR CR SR seq H heads B backwards

theorem source_split (candidate : Bool) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    (source candidate Q C S QR CR SR seq H heads B backwards).surfaceBody =
      (Kernels.rope Q C S QR CR SR seq H heads B backwards).surfaceBody.take 9 ++
      [headStep candidate] ++
      (Kernels.rope Q C S QR CR SR seq H heads B backwards).surfaceBody.drop 10 := by
  cases candidate <;> cases backwards <;> rfl

theorem head_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    step M (headStep Bool.false) s = step M (headStep Bool.true) s := by
  simp only [headStep, Bool.false_eq_true, ↓reduceIte, step, evalExpr, evalOp]
  cases ha : s.regs .nat [] "group_head_position" <;>
    cases hb : s.regs .nat [] "ROPE_GROUP_SIZE" <;> simp_all [numeric]
  apply congrArg (fun v => s.setReg "head_start" .nat [] v)
  funext i
  exact Nat.mul_comm _ _

theorem head_run {α : Type} [Inhabited α] (M : Algebra α) :
    run M [headStep Bool.false] = run M [headStep Bool.true] := by
  funext s
  simp only [run,head_step]

theorem head_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (candidate : Bool) (s : State α) :
    Safety.traceSafeList M bounds [headStep candidate] s := by
  rw [Safety.traceSafeList]
  constructor
  · cases candidate <;> simp [headStep,Safety.stepSafe,Safety.safeExpr,Safety.safeOp]
  · cases step M (headStep candidate) s <;> simp [Safety.traceSafeList]

theorem source_exec {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    exec M (Kernels.rope Q C S QR CR SR seq H heads B backwards) s =
      exec M (Kernels.ropeIndexCommuted Q C S QR CR SR seq H heads B backwards) s := by
  change run M (source Bool.false Q C S QR CR SR seq H heads B backwards).surfaceBody s =
    run M (source Bool.true Q C S QR CR SR seq H heads B backwards).surfaceBody s
  simp only [source_split,run_append,head_run]

/-- The changed statement accesses no memory; its equal successor preserves
all later masked-load/store safety checks, including failures. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    TraceSafe M bounds (Kernels.rope Q C S QR CR SR seq H heads B backwards) s ↔
      TraceSafe M bounds (Kernels.ropeIndexCommuted Q C S QR CR SR seq H heads B backwards) s := by
  change Safety.traceSafeList M bounds
      (source Bool.false Q C S QR CR SR seq H heads B backwards).surfaceBody s ↔
    Safety.traceSafeList M bounds
      (source Bool.true Q C S QR CR SR seq H heads B backwards).surfaceBody s
  simp only [source_split,Safety.traceSafeList_append,run_append,head_run,head_safe]

end
end VeriTile.Bench.Repairs.FastRopeEmbedding.SourcePair
