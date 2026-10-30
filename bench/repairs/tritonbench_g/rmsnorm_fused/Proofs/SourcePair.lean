import bench.repairs.tritonbench_g.rmsnorm_fused.PairIO
import bench.tritonbench_g.l2_norm_triton1.Proofs.FP

namespace VeriTile.Bench.Repairs.RmsnormFused.SourcePair
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- The repaired PID is signed integer data. Preserve its conversion and
commute only the two factors before the existing pointer conversion. -/
@[spec_rule] theorem output_pointer_commutes {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) (before : List ComputeStmt)
    (Y : RegionName) (N : Nat) :
    OpEquivalent M s before
      (.ptrAdd .nil (.ptrBase Y) (.castIntToNat
        (.mul .int .nil (.ref .int [] "row") (.castNatToInt (.constNat N)))))
      (.ptrAdd .nil (.ptrBase Y) (.castIntToNat
        (.mul .int .nil (.castNatToInt (.constNat N)) (.ref .int [] "row")))) := by
  intro t _
  simp only [FP.Structural.evalOp,FP.Structural.numeric]
  cases t.regs .int [] "row" <;>
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]

/-- The existing exact address rewrite applies to the repaired runtime body;
the floating operations are opaque and receive exactly the same inputs. -/
theorem equivalent (X Y W : RegionName) (N B : Nat) :
    original X Y W N B ≡[assumptions] addressCommuted X Y W N B := by
  equiv_decompose
  fp_prove

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RmsnormFused.SourcePair
