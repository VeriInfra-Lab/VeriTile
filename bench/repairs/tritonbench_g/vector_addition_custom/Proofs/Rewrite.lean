import bench.repairs.tritonbench_g.vector_addition_custom.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Value
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
open FP.ProfiledRewrite (engine)
open FP.ProfiledMemory (Site)
noncomputable section
set_option maxRecDepth 24000

/-- Execute the original source with both explicit FP32 load boundaries. -/
def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (add_kernel "x" "y" "out" n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [add_kernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, Option.bind, numeric, numericLt, natLt, bop, store, Region.cast]
  rfl

def commutedExecution {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (add_commuted_kernel "x" "y" "out" n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [add_commuted_kernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, Option.bind, numeric, numericLt, natLt, bop, store, Region.cast]
  rfl

/-- The actual store mask and operands after the original load prefix. -/
def additionSite (n B : Nat) : Site :=
  .maskedBinaryFP32 (VeriTile.Triton.Source.prefixBeforeStore (add_kernel "x" "y" "out" n B).surfaceBody)
    [B] (.lt .nat .scalarR (.ref .nat [B] "offs") (.constNat n)) (.ref .real [B] "a") (.ref .real [B] "b")

/-- Actual source load result; masked values remain arbitrary. -/
def loadedValue {α : Type} [Inhabited α] (M : Algebra α) (region : RegionName)
    (n B : Nat) (s : State α) (i : TileIndex [B]) : α :=
  M.fp32Load (if s.pids 0 * B + i.1.val < n then
    (s.mem region (s.pids 0 * B + i.1.val)).read .real
    else s.undef .real region (s.pids 0 * B + i.1.val))

theorem site_values_iff {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (n B : Nat) (s : State α) :
    (additionSite n B).Holds M D s ↔
      ∀ i : TileIndex [B], s.pids 0 * B + i.1.val < n →
        D .finite (loadedValue M "x" n B s i) ∧ D .finite (loadedValue M "y" n B s i) := by
  simp [additionSite, Site.Holds, VeriTile.Triton.Source.prefixBefore,
    VeriTile.Triton.Source.prefixBeforeStore, add_kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind, numeric,
    numericLt, natLt, bop, Region.cast, loadedValue,
    engine, Algebra.withDefaultPrecision, resolvePrecision]

/-- Strict FP32 addition commutation is used only on active lanes.
Inactive lanes do not store. -/
@[spec_rule] theorem commuted_memory {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (hM : Models [] M D) (n B : Nat) (s : State α)
    (hd : (additionSite n B).Holds M D s) :
    (originalExecution M n B s).val.mem = (commutedExecution M n B s).val.mem := by
  have finite := (site_values_iff M D n B s).mp hd
  have hvalues (i : TileIndex [B]) (hi : s.pids 0 * B + i.1.val < n) :
      M.binary (some .fp32) .real .add (loadedValue M "x" n B s i)
        (loadedValue M "y" n B s i) =
      M.binary (some .fp32) .real .add (loadedValue M "y" n B s i)
        (loadedValue M "x" n B s i) :=
    FP.ScalarArithmetic.add_comm_strict M D hM s _ _ (finite i hi).1 (finite i hi).2
  simp only [originalExecution, commutedExecution, bop, Broadcast.leftIndex,
    Broadcast.rightIndex, decide_eq_true_eq,
    VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
    State.setReg_mem]
  simp only [engine, Algebra.withDefaultPrecision, resolvePrecision,
    Option.getD_some, Option.getD_none]
  simp only [loadedValue] at hvalues
  congr 1
  funext mem i
  by_cases hi : s.pids 0 * B + i.1.val < n
  · simp only [hi, ↓reduceIte]
    have h := hvalues i hi
    simp only [hi, ↓reduceIte] at h
    rw [h]
  · simp only [hi, ↓reduceIte]

#axiomsClean originalExecution
#axiomsClean commutedExecution
#axiomsClean site_values_iff
#axiomsClean commuted_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Rewrite
