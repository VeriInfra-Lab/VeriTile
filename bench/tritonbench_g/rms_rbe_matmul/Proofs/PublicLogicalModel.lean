import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicIOLayout

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

def coordinate (B : Nat) (p : Parameters) (j : TileIndex [B,p.M,p.N]) : KernelGrid.Coordinate B p :=
  (j.1,j.2.1,j.2.2.1)

/-- Independent source-ordered projection/optional rotary expression for the
whole logical output tensor, including its selected final storage cast. -/
def formula (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) : ChannelValues α (outputs fp8 B p) :=
  Fin.cases (fun j => ofFloat (outputDType fp8)
    (KernelGraph.output a fp8 p (logical fp8 B p xs)
      (KernelGrid.tile p (KernelGrid.owner p (coordinate B p j)))
      (KernelGrid.lane (coordinate B p j)))) (fun k => k.elim0)

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGrid.launch]
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
