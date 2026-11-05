import bench.tritonbench_g.rms_rbe_matmul.Proofs.LogicalReader
import VeriTile.Meta.Specification
import bench.tritonbench_g.rms_rbe_matmul.Proofs.RotaryWhole
import bench.tritonbench_g.rbe_triton_transform.RealCorrect

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

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.RotaryCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
open RbeTritonTransformCheckedLaunch
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The complete separately pasted rotary function, with its original dense
2-by-1024 launch, odd-tail padding, both barriers, execution, safety and frame. -/
specification rotary_real (r : Request) (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    Spec.Real (RotaryIO.logical r X Y qs ⊨ fun xs b m c =>
      let n := c.val/2*2
      let u := LogicalView.column r xs b m n
      let v := LogicalView.column r xs b m (n+1)
      let angle := ((r.position:ℝ)+(m.val:ℝ)) /
        Real.rpow 10000 ((n%128:Nat)/(128:ℝ))
      if decide (c.val%2=1) then u*Real.sin angle+v*Real.cos angle
      else u*Real.cos angle-v*Real.sin angle) := by
  apply RotaryBridge.implements
  exact RbeTritonTransformCheckedLaunch.Correct.kernel_correct r X Y different qs all

#axiomsClean rotary_real
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.RotaryCorrect
