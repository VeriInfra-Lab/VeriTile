import bench.tritonbench_g.uniform_sampling.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.UniformSampling.Correct
open VeriTile.Triton FP.Structural KernelMath
open VeriTile.Bench.TritonBenchG.UniformSampling
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Real interpretation of the shared field/lane interface in IO.lean. -/
abbrev samplingIO := KernelPublic.logicalIO realAlgebra

/-- Canonical Philox10 word and its actual rounded uniform decoder, followed
by the independent affine Real formula. -/
def formula (seed offset : Int) (lower upper : ℝ) (pid B : Nat) (f : Fin 4) (i : Fin B) : ℝ :=
  lower + Random.uniformReal (ScalarSpec.word seed offset (pid*B+i.val)
    (KernelExecution.fieldOf f)) * (upper-lower)

/-- Every active field/lane has the formula; the actual source is safe and
terminates, and the complete exterior frame is preserved. Finite denotation
interprets the scalar ABI words in Real; it is not a Python guard. -/
specification uniform_sampling_kernel_correctness (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (lowerValue upperValue : ℝ)
    (lowerDenotation : Denotes lower lowerValue) (upperDenotation : Denotes upper upperValue)
    (B : Nat) (hB : Random.ArangeSizeDomain B)
    (pid : Nat) (_nonempty : 0<N) (block : B=512 ∨ B=1024)
    (_launched : pid<SourceDomain.grid N B) (seedABI : Random.IntegerType.i64.inRange seed=Bool.true)
    (_offsetABI : Random.IntegerType.i64.inRange offset=Bool.true) :
    Spec.Real (samplingIO y N seed offset lower upper B hB pid ⊨ fun _ f i =>
      lowerValue + Random.uniformReal (ScalarSpec.word seed offset (pid*B+i.val)
        (KernelExecution.fieldOf f)) * (upperValue-lowerValue)) := by
  apply (KernelPublic.logical_iff _ _ _ _ _ _ _ _ _ _ _).2
  exact KernelPublic.implements_real y N seed offset lower upper lowerValue upperValue
    lowerDenotation upperDenotation B hB pid (by rcases block with rfl|rfl <;> decide) seedABI

#specNonCircular formula avoiding [kernel,indexCommutedKernel,KernelExecution.result]
#axiomsClean uniform_sampling_kernel_correctness
#stmtSurfaceSubset uniform_sampling_kernel_correctness ⊆
  [RegionName,Float32Bits,Denotes,Random.ArangeSizeDomain,SourceDomain.grid,
   Random.IntegerType,Random.IntegerType.i64,Random.IntegerType.inRange,
   Spec.Real,samplingIO,LogicalProgramIO.Implements,Random.uniformReal,ScalarSpec.word,KernelExecution.fieldOf]
end
end VeriTile.Bench.Repairs.UniformSampling.Correct
