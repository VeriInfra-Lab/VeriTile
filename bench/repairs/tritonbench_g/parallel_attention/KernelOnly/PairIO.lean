import bench.repairs.tritonbench_g.parallel_attention.Proofs.FusedKernel
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
noncomputable section
variable {α : Type} [Inhabited α]

def forwardJob (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) : Invocation α :=
  { IOData.forwardJob M D shape scale p with kernel:=forward (forwardConfig shape scale) }

def queryJob (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) : Invocation α :=
  { IOData.queryJob M D shape scale p with kernel:=query (dqConfig shape scale p.tile) }

/-- This is the retained per-stage compiler model, not separate physical launches.
The unchanged DKV stage follows DQ at every original grid point. -/
def jobs (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    List (Invocation NativeStage.Expr) := match direction with
  | .forward => (programs d.shape 128).map (forwardJob (NativeStage.algebra .forward d)
      (NativeStage.dots .forward d.shape) d.shape scale)
  | .backward => (programs d.shape 64).flatMap fun p=>
      [queryJob (NativeStage.algebra .query d) (NativeStage.dots .query d.shape) d.shape scale p,
       IOData.keyValueJob (NativeStage.algebra .keyValue d) (NativeStage.dots .keyValue d.shape) d.shape scale p]

abbrev io (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :=
  dataIO NativeStage.Expr direction d.shape (nativeJobs direction d scale)

def variant (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    ProgramIO.Variant (io direction d scale) where
  schedule:=fun _=>jobs direction d scale
  observe:=fun _ _=>id

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
