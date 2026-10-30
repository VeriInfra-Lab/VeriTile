import bench.repairs.tritonbench_g.int4_matmul.IO
import VeriTile.Triton.Float.PerAddressProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.Int4Matmul.FoldStrideIO
open VeriTile Triton FP.Structural KernelExecution
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α) (p : IO.Program) :
    Invocation (SourceGraph.Node α) :=
  { IO.invocation x t o rest p with kernel := KernelSource.foldStrideKernel (Raw.parameters x t) }

def variant (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α) (lanes : List Lane) :
    PerAddressProgramIO.Variant (IO.programIO x t o rest lanes) where
  invocation := fun _=>invocation x t o rest
  observe := fun _ _=>SourceGraph.observe o

def original (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α) (lanes : List Lane) :=
  PerAddressProgramIO.Implementation.original (IO.programIO x t o rest lanes) (IO.domain x.shape o)
def optimized (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α) (lanes : List Lane) :=
  PerAddressProgramIO.Implementation.ofVariant (variant x t o rest lanes) (IO.domain x.shape o)



end
end VeriTile.Bench.Repairs.Int4Matmul.FoldStrideIO
