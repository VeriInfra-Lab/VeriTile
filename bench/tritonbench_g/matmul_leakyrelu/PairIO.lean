import bench.tritonbench_g.matmul_leakyrelu.Proofs.Pair
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourcePair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

structure IOProgram (p : KernelInterface.Parameters) (order : Schedule p.request) where
  kernel : ComputeKernel
  fused : Bool

def programIO (cfg : FP.Config) (rest : Algebra Pair.Word)
    (D : FPValue.Dot Pair.Word) (F : FPFused.DotAcc Pair.Word)
    {p : KernelInterface.Parameters} {order : Schedule p.request} (program : IOProgram p order) :
    ProgramIO Pair.Word Pair.Word :=
  let M := Pair.fp32Algebra cfg rest
  { Original.io M D p order with schedule := fun _ => order.order.map (fun pid =>
      { Original.job M D p pid with
        kernel := program.kernel
        backend := if program.fused then FPFused.backend M F else FPValue.backend M D }) }

def variant (cfg : FP.Config) (rest : Algebra Pair.Word)
    (D : FPValue.Dot Pair.Word) (F : FPFused.DotAcc Pair.Word)
    {p : KernelInterface.Parameters} {order : Schedule p.request}
    (left right : IOProgram p order) : ProgramIO.Variant (programIO cfg rest D F left) where
  schedule := (programIO cfg rest D F right).schedule
  observe := fun _ _ => id

/-- The two actual shared ordinary kernels, including their distinct typed
primitives, are retained as program data in the public equivalence statement. -/
def original (p : KernelInterface.Parameters) (order : Schedule p.request) : IOProgram p order :=
  ⟨sourceDsl p.request,Bool.false⟩
def contracted (p : KernelInterface.Parameters) (order : Schedule p.request) : IOProgram p order :=
  ⟨dsl p.request,Bool.true⟩

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourcePair
