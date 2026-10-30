import bench.tritonbench_g.fused_activation.Proofs.Source
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.FusedActivation.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded FP.ProfiledMemory
open FP.ProfiledRewrite (engine)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The actual outer addition operands after all three typed FP32 loads. -/
def additionSite (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) : Site :=
  .maskedBinaryFP32 (Source.prefixBefore (candidateKernel p W m a).surfaceBody "activ_input")
    [p.block] (.ref .bool [p.block] "mask")
    (.add .real (.consSame .nil)
      (.mul .real .scalarL (.const m) (.ref .real [p.block] "tmp3"))
      (.ref .real [p.block] "tmp0"))
    (.ref .real [p.block] "tmp1")

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (address : Nat) (j : TileIndex [p.block]) : α :=
  M.fp32Load (if s.pids 0 * p.block + j.1.val < p.n then
    (s.mem r address).read .real else s.undef .real r address)

def combined {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (m : ℝ) (s : State α) (j : TileIndex [p.block]) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real m)
      (loaded M p s p.x (s.pids 0 * p.block + j.1.val) j))
    (loaded M p s p.output (s.pids 0 * p.block + j.1.val) j)

theorem site_values_iff {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) (s : State α) :
    (additionSite p W m a).Holds M D s ↔
      ∀ j : TileIndex [p.block], s.pids 0 * p.block + j.1.val < p.n →
        D .finite (combined M p m s j) ∧
        D .finite (loaded M p s p.y ((s.pids 0 * p.block + j.1.val) % W) j) := by
  cases a <;> simp [additionSite, Site.Holds, Source.prefixBefore, candidateKernel, kernel,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, numeric, numericLt, natLt, integralMod, bop, Region.cast,
    engine, Algebra.withDefaultPrecision, resolvePrecision, loaded, combined]

def commutedExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) (s : State α) :
    {t : State α // FP.Structural.exec M (commutedKernel p.output p.y p.x W p.n p.block m a) s = some t} := by
  cases a
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [commutedKernel, FP.Structural.exec, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType]
    rfl

def commutedScalar {α : Type} (M : Algebra α) (m : ℝ) (a : Bool) (x bias input : α) : α :=
  let z := M.binary (some .fp32) .real .add bias
    (M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real m) input) x)
  if a then M.unary (some .fp32) .sigmoid z
  else M.binary (some .fp32) .real .max (M.literal (some .fp32) .real 0) z

def commutedCell {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (W : Nat) (m : ℝ) (a : Bool) (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk .real (commutedScalar M m a (M.fp32Load ((mem p.output o).read .real))
    (M.fp32Load ((mem p.y (o%W)).read .real)) (M.fp32Load ((mem p.x o).read .real)))

def commutedMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (W : Nat) (m : ℝ) (a : Bool) (s : State α) :=
  (TileShape.allIndices [p.block]).foldl (fun mem j =>
    if active p s j then fun r o =>
      if r=p.output ∧ o=VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset p s j
      then commutedCell M p W m a s.mem
        (VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset p s j) else mem r o
    else mem) s.mem

theorem commuted_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) (s : State α) :
    (commutedExecution M p W m a s).val.mem = commutedMemory M p W m a s := by
  cases a
  all_goals
    simp [commutedExecution, commutedMemory,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset, active,
      commutedCell, commutedScalar, bop, Broadcast.leftIndex, Broadcast.rightIndex,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory, State.setReg]
    congr 1
    funext mem j
    by_cases bound : s.pids 0 * p.block + j.1.val < p.n
    · simp [bound, numeric, ofFloat, toFloat, integralMod]
    · simp [bound]


theorem commuted_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) (s : State α) (r : RegionName) (o : Nat)
    (outside : r ≠ p.output ∨ ∀ j, active p s j →
      o ≠ VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset p s j) :
    (commutedExecution M p W m a s).val.mem r o = s.mem r o := by
  rw [commuted_memory]
  have scatter : commutedMemory M p W m a s =
      ((TileShape.allIndices [p.block]).foldl (fun t j => if active p s j then
        t.write p.output (VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset p s j)
          (commutedCell M p W m a s.mem
            (VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset p s j)) else t) s).mem := by
    rw [VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory]
    rfl
  rw [scatter]
  apply State.masked_scatter_frame
  exact outside.imp id (fun h j _ hj => h j hj)

/-- Successful source executions agree on every memory cell. Only the final
addition is commuted; activation and inner expression remain unchanged. -/
@[spec_rule] theorem memory_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (laws : Models [] M D)
    (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) (s : State α)
    (domain : (additionSite p W m a).Holds M D s) :
    (execution (engine M) p W m a s).val.mem =
      (commutedExecution (engine M) p W m a s).val.mem := by
  have finite := (site_values_iff M D p W m a s).mp domain
  have adds (j : TileIndex [p.block]) (hj : s.pids 0 * p.block + j.1.val < p.n) :
      M.binary (some .fp32) .real .add (combined M p m s j)
          (loaded M p s p.y ((s.pids 0 * p.block + j.1.val) % W) j) =
        M.binary (some .fp32) .real .add
          (loaded M p s p.y ((s.pids 0 * p.block + j.1.val) % W) j)
          (combined M p m s j) :=
    FP.ScalarArithmetic.add_comm_strict M D laws s _ _ (finite j hj).1 (finite j hj).2
  cases a
  all_goals
    simp only [execution, commutedExecution, bop, Broadcast.leftIndex, Broadcast.rightIndex,
      decide_eq_true_eq, VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
      State.setReg_mem]
    simp only [engine, Algebra.withDefaultPrecision, resolvePrecision,
      Option.getD_some, Option.getD_none]
    congr 1
    funext mem j
    by_cases hj : s.pids 0 * p.block + j.1.val < p.n
    · simp only [hj, ↓reduceIte]
      have eq := adds j hj
      simp only [combined, loaded, hj, ↓reduceIte] at eq
      simp only [numeric, integralMod, ofFloat, toFloat, Option.getD_some] at *
      rw [eq]
    · simp only [hj, ↓reduceIte]

/-- Bundle source success with the proved full-memory rewrite. -/
theorem runs_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (laws : Models [] M D)
    (p : Parameters) (W : Nat) (m : ℝ) (a : Bool) (s : State α)
    (domain : (additionSite p W m a).Holds M D s) :
    ∃ first second, FP.Structural.exec (engine M) (candidateKernel p W m a) s = some first ∧
      FP.Structural.exec (engine M) (commutedKernel p.output p.y p.x W p.n p.block m a) s = some second ∧
      first.mem = second.mem :=
  ⟨(execution (engine M) p W m a s).val, (commutedExecution (engine M) p W m a s).val,
    (execution (engine M) p W m a s).property, (commutedExecution (engine M) p W m a s).property,
    memory_equiv M D laws p W m a s domain⟩

#axiomsClean runs_equiv
#axiomsClean site_values_iff
#axiomsClean commutedExecution
#axiomsClean memory_equiv
end
end VeriTile.Bench.Repairs.FusedActivation.Rewrite
