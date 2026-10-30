import bench.repairs.tritonbench_g.seeded_dropout.IO
import bench.repairs.tritonbench_g.seeded_dropout.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.SeededDropout.KernelPublic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SeededDropout
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Full old/new judgment equivalence, including the original masked memory
binding, source schedule, access safety, successful execution and exact frame. -/
theorem logical_iff (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ)
    (seed : SeedCarrier k) (pid : Nat) (f : Nat → ℝ → ℝ) :
    (logicalIO Whole.realAlgebra k X O n p seed pid ⊨
      fun xs i => f (pid*1024+i.val) (xs i)) ↔
    (io Whole.realAlgebra k X O n p seed pid).Implements
      (target Whole.realAlgebra k X O n p seed pid f) := Iff.rfl

theorem logical_real (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ)
    (seed : SeedCarrier k) (pid : Nat) (hn : n<2^31)
    (hp : pid<gridCount n) (hs : SeedABI k seed) :
    logicalIO Whole.realAlgebra k X O n p seed pid ⊨ fun xs i =>
      if Random.randReal (seedValue k seed) ((pid*1024+i.val : Nat) : Int) .bits32 10 > p
      then xs i/(1-p) else 0 := by
  apply (logical_iff k X O n p seed pid
    (fun i x => if Random.randReal (seedValue k seed) (i : Int) .bits32 10 > p
      then x/(1-p) else 0)).mpr
  apply implements Whole.realAlgebra (fun a b => decide (a<b)) rfl
    k X O n p seed pid hn hp hs
  intro i x
  exact Whole.scalar_real p (seedValue k seed) i x

#axiomsClean logical_iff
#axiomsClean logical_real
end
end VeriTile.Bench.Repairs.SeededDropout.KernelPublic
