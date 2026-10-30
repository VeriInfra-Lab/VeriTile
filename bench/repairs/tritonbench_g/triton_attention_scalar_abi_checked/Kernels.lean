import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.SourceKernels
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardReference
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardReference
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPPreprocess
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Geometry

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI
open VeriTile Triton Source

abbrev Shape := VeriTile.Bench.Repairs.TritonAttention.Whole001.Shape

def forwardParameters (c : Shape) (scale : ℝ) : ForwardParameters :=
  ⟨.fp32,"Q","K","V","L","M","Out",scale,
    c.rows*c.dim,c.rows,c.groups*c.rows,128,128,c.dim⟩
def preParameters (c : Shape) : PreParameters :=
  ⟨.fp32,"Out","DO","L","NewDO","Delta",128,c.dim⟩
def backwardParameters (c : Shape) (scale : ℝ) : BackwardParameters :=
  ⟨.fp32,"Q","K","V","NewDO","DQ","DK","DV","M","Delta",scale,
    c.heads*c.rows*c.dim,c.rows*c.dim,c.heads,c.groups*c.rows,128,c.dim,c.blocks⟩

def forwardKernel (c : Shape) (scale : ℝ) := forwardSource (forwardParameters c scale)
def preprocessKernel (c : Shape) := preSource (preParameters c)
def backwardKernel (c : Shape) (scale : ℝ) := backwardSource (backwardParameters c scale)

end VeriTile.Bench.Repairs.TritonAttentionScalarABI
