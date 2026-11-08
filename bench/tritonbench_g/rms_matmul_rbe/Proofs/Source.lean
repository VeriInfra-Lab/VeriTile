import bench.tritonbench_g.rms_matmul_rbe.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

abbrev Arguments := TypedSource.Arguments

/-- The wrapper always selects 16×64×64 with four warps and four stages. -/
def source (fp8 outputFP8 : Bool) (p : Arguments) : ComputeKernel :=
  TypedSource.kernel fp8 outputFP8 {p with BM:=16,BN:=64,BK:=64}

def prelude (fp8 outputFP8 : Bool) (p : Arguments) :=
  (source fp8 outputFP8 p).surfaceBody.take 24
def body (fp8 outputFP8 : Bool) (p : Arguments) : List ComputeStmt :=
  match ((source fp8 outputFP8 p).surfaceBody[24]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ statements) => statements
  | _ => []
def epilogue (fp8 outputFP8 : Bool) (p : Arguments) :=
  (source fp8 outputFP8 p).surfaceBody.drop 25

def startExpr (p : Arguments) : Op .int [] :=
  match ((source Bool.false Bool.false p).surfaceBody[24]? : Option ComputeStmt) with
  | some (.forRangeInt _ start _ _ _) => start
  | _ => .constInt 0

def countExpr (p : Arguments) : Op .int [] :=
  match ((source Bool.false Bool.false p).surfaceBody[24]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ stop _ _) => stop
  | _ => .constInt 0

def strideExpr (p : Arguments) : Op .nat [] :=
  match ((source Bool.false Bool.false p).surfaceBody[24]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ stride _) => stride
  | _ => .constNat 0

theorem partition (fp8 outputFP8 : Bool) (p : Arguments) :
    (source fp8 outputFP8 p).surfaceBody=prelude fp8 outputFP8 p ++
      [.forRangeInt "step" (startExpr p) (countExpr p) (strideExpr p)
        (body fp8 outputFP8 p)] ++ epilogue fp8 outputFP8 p := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem source_length (fp8 outputFP8 : Bool) (p : Arguments) :
    (source fp8 outputFP8 p).surfaceBody.length=34 := by
  cases fp8 <;> cases outputFP8 <;> rfl

#axiomsClean partition
#axiomsClean source_length
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
