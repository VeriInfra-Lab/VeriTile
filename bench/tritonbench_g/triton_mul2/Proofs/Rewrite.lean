import bench.tritonbench_g.triton_mul2.Kernels
import bench.tritonbench_g.vector_addition.Sources.Mathematical_Value
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TritonMul2.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
open FP.ProfiledRewrite (engine)
open FP.ProfiledMemory (Site)
noncomputable section
set_option maxRecDepth 24000

/-- Execute the displayed original source, retaining all precision tags. -/
def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (X O : RegionName) (n B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (sourceKernel X O n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [sourceKernel, FP.Structural.exec, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, numericLt, natLt, bop,
    store, Region.cast, ComputeDType.eraseDType]
  rfl

/-- Execute the candidate source with the same load and store precision. -/
def commutedExecution {α : Type} [Inhabited α] (M : Algebra α) (X O : RegionName) (n B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (commutedSourceKernel X O n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [commutedSourceKernel, FP.Structural.exec, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, numericLt, natLt, bop,
    store, Region.cast, ComputeDType.eraseDType]
  rfl

/-- Check only lanes selected by the source's actual store mask, after the
original load prefix. The memory proof below establishes masked-lane irrelevance. -/
def multiplicationSite (X O : RegionName) (n B : Nat) : Site :=
  .maskedScalarFP32 (VeriTile.Triton.Source.prefixBefore (sourceKernel X O n B).surfaceBody
    "output") [B] (.ref .bool [B] "mask")
    (.ref .real [B] "x") (.const 2)

/-- The actual load result, including arbitrary values in masked lanes. -/
def loadedValue {α : Type} [Inhabited α] (M : Algebra α) (X O : RegionName) (n B : Nat)
    (s : State α) (i : TileIndex [B]) : α :=
  M.fp32Load (if s.pids 0 * B + i.1.val < n then
    (s.mem X (s.pids 0 * B + i.1.val)).read .real
    else s.undef .real X (s.pids 0 * B + i.1.val))

/-- Evaluate the actual site condition; this is source execution, not an
extra input restriction or an experimental primitive-conformance premise. -/
theorem site_values_iff {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (X O : RegionName) (n B : Nat) (s : State α) :
    (multiplicationSite X O n B).Holds M D s ↔
      ∀ i : TileIndex [B], s.pids 0 * B + i.1.val < n →
        D .finite (loadedValue M X O n B s i) ∧
        D .finite (M.literal (some .fp32) .real 2) := by
  simp [multiplicationSite, Site.Holds, VeriTile.Triton.Source.prefixBefore,
    sourceKernel, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, numeric, numericLt, natLt, bop, Region.cast, loadedValue,
    engine, Algebra.withDefaultPrecision, resolvePrecision]

/-- Complete memory equality for the two displayed kernels, using only
the shared strict FP32 multiplication commutation theorem. -/
@[spec_rule] theorem commuted_memory {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (hM : Models [] M D) (X O : RegionName) (n B : Nat) (s : State α)
    (hd : (multiplicationSite X O n B).Holds M D s) :
    (originalExecution M X O n B s).val.mem = (commutedExecution M X O n B s).val.mem := by
  have finite := (site_values_iff M D X O n B s).mp hd
  have hvalues (i : TileIndex [B]) (hi : s.pids 0 * B + i.1.val < n) :
      M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 2)
        (loadedValue M X O n B s i) =
      M.binary (some .fp32) .real .mul (loadedValue M X O n B s i)
        (M.literal (some .fp32) .real 2) :=
    FP.ScalarArithmetic.mul_comm_strict M D hM s _ _ (finite i hi).2 (finite i hi).1
  simp only [originalExecution, commutedExecution,
    VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
    State.setReg_mem]
  simp only [engine, Algebra.withDefaultPrecision, resolvePrecision,
    Option.getD_some]
  simp only [loadedValue] at hvalues
  congr 1
  funext mem i
  by_cases hi : s.pids 0 * B + i.1.val < n
  · simp only [hi, ↓reduceIte]
    have h := hvalues i hi
    simp only [hi, ↓reduceIte] at h
    rw [h]
  · simp only [hi, ↓reduceIte]

/-- The FP execution preserves every cell outside its active output writes.
This statement has no finiteness or numerical rewrite premise. -/
theorem original_frame {α : Type} [Inhabited α] (M : Algebra α)
    (X O : RegionName) (n B : Nat) (s : State α) (r : RegionName) (address : Nat)
    (outside : r ≠ O ∨ ∀ i : TileIndex [B], s.pids 0 * B + i.1.val < n →
      address ≠ s.pids 0 * B + i.1.val) :
    (originalExecution M X O n B s).val.mem r address = s.mem r address := by
  simp only [originalExecution]
  apply State.masked_scatter_frame
  exact outside.imp id (fun h i _ hi => h i hi)

#axiomsClean originalExecution
#axiomsClean commutedExecution
#axiomsClean site_values_iff
#axiomsClean commuted_memory
#axiomsClean original_frame
end
end VeriTile.Bench.Repairs.TritonMul2.Rewrite
