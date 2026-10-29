import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Kernels
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
  (source fp8 outputFP8 p).surfaceBody.take 12
def body (fp8 outputFP8 : Bool) (p : Arguments) : List ComputeStmt :=
  match ((source fp8 outputFP8 p).surfaceBody[12]? : Option ComputeStmt) with
  | some (.forRangeDyn _ _ _ _ statements) => statements
  | _ => []
def epilogue (fp8 outputFP8 : Bool) (p : Arguments) :=
  (source fp8 outputFP8 p).surfaceBody.drop 13

def countExpr (p : Arguments) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil
    (.add .nat .nil (.constNat p.K) (.constNat 64)) (.constNat 1)) (.constNat 64)

theorem partition (fp8 outputFP8 : Bool) (p : Arguments) :
    (source fp8 outputFP8 p).surfaceBody=prelude fp8 outputFP8 p ++
      [.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1)
        (body fp8 outputFP8 p)] ++ epilogue fp8 outputFP8 p := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem source_length (fp8 outputFP8 : Bool) (p : Arguments) :
    (source fp8 outputFP8 p).surfaceBody.length=22 := by
  cases fp8 <;> cases outputFP8 <;> rfl

#axiomsClean partition
#axiomsClean source_length
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
