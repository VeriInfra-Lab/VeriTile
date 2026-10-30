import bench.repairs.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.chunk_gated_attention.Proofs.FP

/-! Complete FP source equivalence for the actual repaired typed kernels.
The old generic finite-site requirements and scalar dot operand descriptions
are reused; the changed pointer and decoded-grid sources are proved here. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
set_option maxHeartbeats 16000000
set_option maxRecDepth 20000

/-- The repaired block pointers have static zero offsets. -/
@[spec_rule] theorem static_pointer_base {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (before : List ComputeStmt)
    (r : RegionName) (a b : Op .nat []) (parent block : TileShape)
    (strides offsets : List Nat) (h : OpEquivalent M s before a b) :
    OpEquivalent M s before (.makeBlockPtrDyn r a parent block strides offsets)
      (.makeBlockPtrDyn r b parent block strides offsets) := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [h t ht]

/-- Unconditional semantic equality lifts through the batch-exhaustion guard. -/
theorem contextual_if_unconditional {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (condition : Op .bool [])
    (lhs rhs : List ComputeStmt)
    (hbody : ∀ t, run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThen condition lhs) (.ifThen condition rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : FP.Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | «false» => rfl
      | «true» => simp only [↓reduceIte]; rw [hbody t]

theorem nat_mul_const (v : TileIndex [] → Nat) (n : Nat) :
    FP.Structural.bop Nat.mul .nil v (fun _ => n) =
      FP.Structural.bop Nat.mul .nil (fun _ => n) v := by
  funext i
  exact Nat.mul_comm _ _

section Cumulative
variable (dtype : FloatDType) (input o : RegionName)
    (head time feature T S BT BS BH base : Nat)
def cumulative : Program where
  kernel := NativeSource.cumOriginal dtype .fp32 input o head time feature T S BT BS BH base
  domain := []
def cumulativeIndexFirst : Program :=
  { cumulative dtype input o head time feature T S BT BS BH base with
    kernel := NativeSource.cumIndexFirst dtype .fp32 input o head time feature T S BT BS BH base }

theorem cumulative_equiv :
    cumulative dtype input o head time feature T S BT BS BH base
      ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    cumulativeIndexFirst dtype input o head time feature T S BT BS BH base := by
  equiv_decompose
  apply contextual_if_unconditional
  intro t
  simp only [run,step,FP.Structural.evalExpr,FP.Structural.evalOp,
    FP.Structural.numeric,FP.Structural.bop,Nat.mul_comm]
  simp only [Option.bind_eq_bind,Option.bind_some,Option.bind_none,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  repeat' first | exact nat_mul_const _ _ | exact Nat.mul_comm _ _ | congr! 1
end Cumulative

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
