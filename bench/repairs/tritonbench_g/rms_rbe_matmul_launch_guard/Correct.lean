import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.LogicalReader
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- RMS-normalized projection over the logical K entries, independent of tiles. -/
def normalizedProjection (K : Nat) (v : KernelMath.Input) (column : Nat) : ℝ :=
  (∑ k ∈ Finset.range K, v.x k*v.rms k*v.weight column k) *
    (1 / Real.sqrt ((∑ k ∈ Finset.range K, v.x k^2)/(K:ℝ)+1/1000000))

/-- Optional adjacent-pair rotary transform with the source's fixed constants. -/
def projectedRotary (K headDim : Nat) (start : Int) (rotary : Bool)
    (v : KernelMath.Input) (row column : Nat) : ℝ :=
  if rotary then
    let even := column-column%2
    let phase := ((start:ℝ)+(row:ℝ)) /
      Real.rpow 10000 (((even%headDim:Nat):ℝ)/(headDim:ℝ))
    if column%2=0 then
      normalizedProjection K v even*Real.cos phase - normalizedProjection K v (even+1)*Real.sin phase
    else
      normalizedProjection K v even*Real.sin phase + normalizedProjection K v (even+1)*Real.cos phase
  else normalizedProjection K v column

specification rms_rbe_matmul_grid_real
    (d8 : Float8E5Bits → ℝ) (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) (evenN : p.N%2=0) :
    Spec.Real (LogicalIO.io d8 fp8 B p ⊨{LogicalIO.domain fp8 B p}
      fun values batch row col =>
        projectedRotary p.K p.headDim p.start p.rotary
          (LogicalIO.rowInput d8 fp8 B p values batch.val row.val) row.val col.val) :=
  (LogicalIO.logical_iff d8 fp8 B p).mpr (PublicIO.real_implements d8 zero fp8 B p evenN)

#stmtSurfaceSubset rms_rbe_matmul_grid_real ⊆
  [Spec.Real,Float8E5Bits,Float8E5Bits.ofInt,KernelSource.Parameters,KernelSource.Parameters.N,
   KernelSource.Parameters.K,KernelSource.Parameters.headDim,KernelSource.Parameters.start,
   KernelSource.Parameters.rotary,LogicalIO.io,LogicalProgramIO.ImplementsOn,
   LogicalIO.domain,LogicalIO.rowInput,LogicalIO.Inputs,KernelSource.Parameters.M,projectedRotary]
#specNonCircular projectedRotary avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGraph.output,KernelGrid.launch]
#axiomsClean rms_rbe_matmul_grid_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard
