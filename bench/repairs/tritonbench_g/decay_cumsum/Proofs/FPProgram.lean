import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Source
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.DecayCumsum.FPProgram
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def prefixCount : Kind → Nat
  | .forward => 7 | .prepare => 10 | .backward => 14

def initCode (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : List ComputeStmt :=
  (Kernels.source kind h p scale).surfaceBody.take (prefixCount kind)

def loopBody (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : List ComputeStmt :=
  match (Kernels.source kind h p scale).surfaceBody.getLast? with
  | some (.forRangeDyn _ _ _ _ body) => body
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

def loopCode (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : ComputeStmt :=
  match kind with
  | .forward | .prepare => .forRangeInt "i" (.constInt 0) ((Op.constNat p.span).castNatToInt) (.constNat 1)
      (loopBody kind h p scale)
  | .backward => .forRangeInt "__rev_j"
      (.sub .int .nil (.constInt 0) (.constInt (Int.ofNat p.span - 1)))
      (.sub .int .nil (.constInt 0) (.sub .int .nil (.constInt 0) (.constInt 1)))
      (.constNat 1) (loopBody kind h p scale)

theorem source_body (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (Kernels.source kind h p scale).surfaceBody=initCode kind h p scale++[loopCode kind h p scale] := by
  cases kind <;> rfl

/-! The inherited source's FP32 storage tag is `.real`, as checked in the
original FPDecayCumsumSource regression. This is the arbitrary FP carrier tag,
not Real arithmetic: all floating callbacks use the selected FP32 profile.
The storage adapter will bind these tagged payloads to actual four-byte FP32.
-/
#axiomsClean source_body
end
end VeriTile.Bench.Repairs.DecayCumsum.FPProgram
