import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_FPReference

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.SourceArithmetic
open VeriTile.Triton FP.Structural

/-- These identities concern only exact integer address/control arithmetic. -/
theorem nat_not_negative (n : Nat) : ¬ (n : Int) < 0 := by omega
theorem add_nat (a b : Nat) : (a : Int) + (b : Int) = ((a+b : Nat) : Int) := by omega
theorem mul_nat (a b : Nat) : (a : Int) * (b : Int) = ((a*b : Nat) : Int) := by exact_mod_cast rfl

theorem ceil_nat (n b : Nat) :
    (if (n : Int) + ((b : Int)-1) < 0 then
      -((1-(b : Int)+ -(n : Int))/(b : Int))
    else ((n : Int)+((b : Int)-1))/(b : Int)) = (((n+b-1)/b : Nat) : Int) := by
  by_cases hb : b=0
  · simp [hb]
  · have h : 1≤n+b := by omega
    have nonnegative : ¬ (n : Int)+((b : Int)-1)<0 := by omega
    rw [if_neg nonnegative]
    have numerator : (n : Int)+((b : Int)-1)=((n+b-1 : Nat) : Int) := by omega
    rw [numerator]
    exact (Int.natCast_ediv _ _).symm

theorem mod_nat (a b : Nat) :
    (a : Int) - (b : Int)*((a : Int)/(b : Int)) = ((a%b : Nat) : Int) := by
  rw [← Int.emod_def]
  exact (Int.natCast_emod _ _).symm

theorem mod_nat_pulled (a b : Nat) :
    (a : Int) - ((b*(a/b) : Nat) : Int) = ((a%b : Nat) : Int) := by
  calc
    _ = (a : Int)-(b : Int)*((a : Int)/(b : Int)) := by simp
    _ = _ := mod_nat a b

theorem mask_nat (lane inner cur width : Nat) :
    (lane : Int) < (inner : Int)-(cur : Int)*(width : Int) ↔ lane < inner-cur*width := by
  rw [mul_nat]
  omega

theorem exec_surface {α : Type} [Inhabited α] (M : Algebra α) (k : ComputeKernel) (s : State α) :
    FP.Structural.exec M k s = run M k.surfaceBody s := by
  cases k
  rfl

theorem stopOp_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : FPValue.Parameters) (s : State α) :
    FP.Structural.evalOp M none (FPValue.stopOp p) s =
      some (fun _ => (((p.inner+p.blockInner-1)/p.blockInner : Nat) : Int)) := by
  simp [FPValue.stopOp,FPValue.source,FPValue.typed,FPValue.commonView,
    VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,
    VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,
    ComputeKernel.surfaceBody,evalOp_unfold,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Option.bind,ceil_nat]

theorem loop_run {α : Type} [Inhabited α] (M : Algebra α) (p : FPValue.Parameters)
    (code : List ComputeStmt) (s : State α) :
    run M [ComputeStmt.forRangeInt "k" (.constInt 0) (FPValue.stopOp p) (.constNat 1) code] s =
      FP.Structural.rangeInt M "k" 0 (((p.inner+p.blockInner-1)/p.blockInner : Nat) : Int) 1 code s := by
  simp only [run,step,stopOp_eval M p s,evalOp_unfold,bind,pure,Option.bind]
  cases FP.Structural.rangeInt M "k" 0 (((p.inner+p.blockInner-1)/p.blockInner : Nat) : Int) 1 code s <;> rfl

#axiomsClean ceil_nat
#axiomsClean mask_nat
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.SourceArithmetic
