import bench.repairs.tritonbench_g.add_value.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Value
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
open FP.ProfiledRewrite (engine)
open FP.ProfiledMemory (Site)
noncomputable section
set_option maxRecDepth 24000

/-- Execute the displayed original source, retaining all precision tags. -/
def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (originalKernel n) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [originalKernel, kernel, FP.Structural.exec, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, numericLt, natLt, bop,
    store, Region.cast]
  rfl

/-- The local constant assignment is part of the candidate's actual execution. -/
def commutedExecution {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (commutedKernel n) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [commutedKernel, FP.Structural.exec, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, numericLt, natLt, bop,
    store, Region.cast]
  rfl

/-- Check only lanes selected by the source's actual store mask, after the
original load prefix. The memory proof below establishes masked-lane irrelevance. -/
def additionSite (n : Nat) : Site :=
  .maskedScalarFP32 (VeriTile.Triton.Source.prefixBefore (originalKernel n).surfaceBody
    "output") [blockSize] (.ref .bool [blockSize] "mask")
    (.ref .real [blockSize] "x") (.const 10)

/-- The actual load result, including arbitrary values in masked lanes. -/
def loadedValue {α : Type} [Inhabited α] (M : Algebra α) (n : Nat)
    (s : State α) (i : TileIndex [blockSize]) : α :=
  M.fp32Load (if s.pids 0 * blockSize + i.1.val < n then
    (s.mem "x" (s.pids 0 * blockSize + i.1.val)).read .real
    else s.undef .real "x" (s.pids 0 * blockSize + i.1.val))

/-- Evaluate the actual site condition; this is source execution, not an
extra input restriction or an experimental primitive-conformance premise. -/
theorem site_values_iff {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (n : Nat) (s : State α) :
    (additionSite n).Holds M D s ↔
      ∀ i : TileIndex [blockSize], s.pids 0 * blockSize + i.1.val < n →
        D .finite (loadedValue M n s i) ∧
        D .finite (M.literal (some .fp32) .real 10) := by
  simp [additionSite, Site.Holds, VeriTile.Triton.Source.prefixBefore,
    originalKernel, kernel, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, numeric, numericLt, natLt, bop, Region.cast, loadedValue,
    engine, Algebra.withDefaultPrecision, resolvePrecision]

/-- Complete memory equality for the two displayed kernels. Local `y` is
executed normally; only strict FP32 addition commutation changes the arithmetic. -/
@[spec_rule] theorem commuted_memory {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (hM : Models [] M D) (n : Nat) (s : State α)
    (hd : (additionSite n).Holds M D s) :
    (originalExecution M n s).val.mem = (commutedExecution M n s).val.mem := by
  have finite := (site_values_iff M D n s).mp hd
  have hvalues (i : TileIndex [blockSize]) (hi : s.pids 0 * blockSize + i.1.val < n) :
      M.binary (some .fp32) .real .add (loadedValue M n s i)
        (M.literal (some .fp32) .real 10) =
      M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 10)
        (loadedValue M n s i) :=
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
  by_cases hi : s.pids 0 * blockSize + i.1.val < n
  · simp only [hi, ↓reduceIte]
    have h := hvalues i hi
    simp only [hi, ↓reduceIte] at h
    rw [h]
  · simp only [hi, ↓reduceIte]

#axiomsClean originalExecution
#axiomsClean commutedExecution
#axiomsClean site_values_iff
#axiomsClean commuted_memory
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.Rewrite
