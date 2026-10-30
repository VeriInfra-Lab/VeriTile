import bench.repairs.tritonbench_g.kldiv_ops.Kernels
import bench.tritonbench_g.kldiv_ops.Kernels

/-! Proof-only decomposition of the complete source in Kernels.lean.
These projections never construct the authoritative executable kernel. -/
namespace VeriTile.Bench.Repairs.KldivOps
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def forwardTemplate (p : Parameters) (flag : Bool) (_eps : ℝ) (red : Nat) : ComputeKernel :=
  KernelDSL.forward p flag red

def epsilonLoss (B : Nat) : ComputeStmt :=
  .assign .real [B] "loss" (.compute (.alg .fp32
    (.mul .real (.consSame .nil) (.ref .real [B] "y_true")
      (.sub .real (.consSame .nil)
        (.log (.maximumFloat .real .scalarR (.ref .real [B] "y_true")
          (.ref .real [] "eps")))
        (.ref .real [B] "y")))))

def forwardPrefix (p : Parameters) (flag : Bool) (red : Nat) :=
  (forwardTemplate p flag 0 red).surfaceBody.take 10

def forwardTemplateBody (p : Parameters) (flag : Bool) (red : Nat) : List ComputeStmt :=
  match ((forwardTemplate p flag 0 red).surfaceBody[10]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

def forwardBody (p : Parameters) (flag : Bool) (red : Nat) : List ComputeStmt :=
  forwardTemplateBody p flag red

def forwardFinish (p : Parameters) (flag : Bool) (red : Nat) :=
  (forwardTemplate p flag 0 red).surfaceBody.drop 11

theorem forward_decomposition (p : Parameters) (flag : Bool) (red : Nat) :
    (forward p flag red).surfaceBody = forwardPrefix p flag red ++
      [.forRangeInt "i" (.constInt 0) (.ref .int [] "n_cols") (.constNat p.block)
        (forwardBody p flag red)] ++ forwardFinish p flag red := by
  cases flag <;> rfl

theorem forward_length (p : Parameters) (flag : Bool) (red : Nat) :
    (forward p flag red).surfaceBody.length=12 := by cases flag <;> rfl

theorem body_length (p : Parameters) (flag : Bool) (red : Nat) :
    (forwardBody p flag red).length=6 := by cases flag <;> rfl

/-- Both target branches use the complete original backward Python body. -/
theorem backward_reuses_complete_source (p : Parameters) (flag : Bool) :
    backward p flag = KernelDSL.backwardBody p flag := by cases flag <;> rfl
end VeriTile.Bench.Repairs.KldivOps
