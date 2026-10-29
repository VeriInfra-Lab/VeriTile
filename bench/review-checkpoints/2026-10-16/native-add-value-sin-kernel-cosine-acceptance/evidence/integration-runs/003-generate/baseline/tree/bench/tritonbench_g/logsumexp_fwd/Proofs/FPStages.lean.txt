import bench.tritonbench_g.logsumexp_fwd.LogsumexpFwd
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

/-- Complete original per-program body, with an explicit fp32 storage profile. -/
def rowBlock (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    ComputeKernel := triton {
  tl.region x = tl.float32, z = tl.float32
  i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
  o_d = i_d * $(B) + tl.arange(0, $(B))
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-float("inf"))
  if HAS_SCALE {
    b_x = b_x * $(scale)
  }
  b_m = tl.max(b_x, 0)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}

def inputAddress {α : Type} (s : State α) (D B : Nat) (i : Fin B) : Nat :=
  s.pids 0 * D + (s.pids 1 * B + i.val)

def outputAddress {α : Type} (s : State α) (D B : Nat) : Nat :=
  s.pids 0 * ((D + B - 1) / B) + s.pids 1

/-- Ordered finite-precision operation graph, including all padded lanes. -/
def rowValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) : α :=
  let loaded : TileIndex [B] → α := fun i => M.fp32Load
    (if s.pids 1 * B + i.1.val < D then
      (s.mem x (inputAddress s D B i.1)).read .real else M.negInf)
  let scaled := if HAS_SCALE then fun i =>
    M.binary (some .fp32) .real .mul (loaded i) (M.literal (some .fp32) .real scale)
    else loaded
  let maximum := M.reduceMax (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false scaled PUnit.unit
  let exponentials := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (scaled i) maximum)
  let total := M.reduceSum (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false exponentials PUnit.unit
  M.binary (some .fp32) .real .add (M.unary (some .fp32) .log total) maximum

specification row_block_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) (s : State α) (hB : 0 < B) :
    ∃ final, FP.Structural.exec M (rowBlock x z D B HAS_SCALE scale) s = some final ∧
      final.mem = (s.write z (outputAddress s D B)
        (.mk .real (rowValue M s x D B HAS_SCALE scale))).mem := by
  have hsum : 1 ≤ D + B := by omega
  cases HAS_SCALE
  all_goals simp [rowBlock, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, rowValue, inputAddress, outputAddress,
    TileShape.allIndices, TileShape.axisDim, hB, State.write]
  all_goals simp only [← Int.natCast_one, ← Int.natCast_sub hsum, ← Int.natCast_add, ← Int.natCast_mul, ← Int.natCast_ediv,
    Int.toNat_natCast, Int.ofNat_lt]
  all_goals rfl

#axiomsClean row_block_fp32
end
end VeriTile.Bench.TritonBenchG.LogsumexpFwd.FPStages
