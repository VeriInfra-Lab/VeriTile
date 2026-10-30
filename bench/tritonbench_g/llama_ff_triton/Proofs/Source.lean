import bench.tritonbench_g.llama_ff_triton.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

structure Parameters where
  rows : Nat
  columns : Nat
  inner : Nat
  byte : Bool
  epsilon : ℝ

def source (p : Parameters) := Source.kernel p.byte p.rows p.columns p.inner p.epsilon
def prelude (p : Parameters) := (source p).surfaceBody.take 25
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody[25]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeInt _ _ _ _ statements) => statements
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 26

def countOp : Op .int [] :=
  .truncDiv .nil (.sub .int .nil (.add .int .nil
    (.ref .int [] "K64") (.constInt 64)) (.constInt 1)) (.constInt 64)

theorem partition (p : Parameters) :
    (source p).surfaceBody = prelude p ++
      [.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (body p)] ++ epilogue p := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8 <;> rfl

/-- The local adapter expands signed cdiv exactly on the positive block
divisors used by the source. No cast or arithmetic index is dropped. -/
theorem signed_cdiv (a b : Nat) (hb : 0 < b) :
    Int.tdiv ((a : Int) + b - 1) b = ((a + b - 1) / b : Nat) := by
  have hc : ((a+b-1 : Nat) : Int) = (a : Int)+b-1 := by omega
  rw [← hc]
  simp

#axiomsClean partition
#axiomsClean signed_cdiv
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
