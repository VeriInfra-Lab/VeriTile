import bench.tritonbench_g.uniform_sampling.Kernels
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_FP

namespace VeriTile.Bench.Repairs.UniformSampling.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def source (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  if swapped then indexCommutedKernel y N seed offset lower upper B hB
  else kernel y N seed offset lower upper B hB

def indexOp (swapped : Bool) (B : Nat) : Op .int [B] :=
  .add .int .scalarL
    (if swapped then .mul .int .nil (.castNatToInt (.ref .nat [] "BLOCK_arg")) (.ref .int [] "pid")
     else .mul .int .nil (.ref .int [] "pid") (.castNatToInt (.ref .nat [] "BLOCK_arg")))
    (.ref .int [B] "lane")

def indexStmt (swapped : Bool) (B : Nat) : ComputeStmt :=
  .assign .int [B] "i4" (.alg (indexOp swapped B))

theorem source_split (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    (source swapped y N seed offset lower upper B hB).surfaceBody =
      (kernel y N seed offset lower upper B hB).surfaceBody.take 11 ++
      indexStmt swapped B :: (kernel y N seed offset lower upper B hB).surfaceBody.drop 12 := by
  cases swapped <;> rfl

theorem index_step (B : Nat) {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    step M (indexStmt Bool.false B) s = step M (indexStmt Bool.true B) s := by
  cases hb : s.regs .nat [] "BLOCK_arg" <;>
    cases hp : s.regs .int [] "pid" <;>
    cases hl : s.regs .int [B] "lane" <;>
    simp [indexStmt, indexOp, step, evalExpr, evalOp_unfold, hb, hp, hl, numeric]
  congr 1
  funext i
  simp [bop, Int.mul_comm]

theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : Float32Bits) (B : Nat)
    (hB : Random.ArangeSizeDomain B) (s : State α) :
    FP.Structural.exec M (kernel y N seed offset lower upper B hB) s =
      FP.Structural.exec M (indexCommutedKernel y N seed offset lower upper B hB) s := by
  change run M (source Bool.false y N seed offset lower upper B hB).surfaceBody s =
    run M (source Bool.true y N seed offset lower upper B hB).surfaceBody s
  rw [source_split, source_split]
  exact TritonBenchG.UniformSampling.FPProof.run_insert M _ _ _ _ (index_step B M) s

#axiomsClean source_split
#axiomsClean index_step
#axiomsClean execution_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelSource
