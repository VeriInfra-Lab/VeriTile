import bench.repairs.tritonbench_g.fast_rope_embedding.IO
import bench.repairs.tritonbench_g.rope_embedding.Proofs.SourceBinding
import VeriTile.Triton.Float.ProfiledRewrite

/-! Complete typed source views for the existing index-commutation pair.
The source precision, both rotation directions, every dynamic loop iteration,
both Q stores and the unrestricted input domain are identical on both sides. -/
namespace VeriTile.Bench.Repairs.RopeEmbedding.PairIO
open VeriTile.Triton FP.ProfiledRewrite

def original (p : Source.Parameters) (back : Bool) : Program :=
  ⟨Source.source p back, []⟩

def indexCommuted (p : Source.Parameters) (back : Bool) : Program :=
  ⟨Kernels.indexCommuted p.input p.cosine p.sine p.rowStride p.cosineStride p.sineStride
    p.length p.dimension p.heads p.width back, []⟩

def assumptions : VeriTile.Spec.Assumptions FP.GuardedFragment := []

end VeriTile.Bench.Repairs.RopeEmbedding.PairIO

namespace VeriTile.Bench.Repairs.RopeEmbedding.Interface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Reuse the single shared rotary memory descriptor with this actual source. -/
def invocation (op : Operations A) (r : Request) (back : Bool) (row group : Nat) : Invocation A where
  kernel := Source.source (parameters r) back
  backend := op.backend
  pids := fun ax => if ax = 0 then row else if ax = 1 then group else 0

def programIO (op : Operations A) (r : Request) (back : Bool) (row group : Nat) : ProgramIO A A :=
  { FastRopeEmbedding.Interface.memory A r row group with
    schedule := fun _ => [invocation op r back row group]
    encode := id
    observe := fun _ _ => id }

end
end VeriTile.Bench.Repairs.RopeEmbedding.Interface

namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton FP.Structural
noncomputable section

/-- Logical rotation coordinates, with the ordinary RoPE source as the core. -/
def ropeIO (r : Request) (back : Bool) (row group : Nat) :
    LogicalProgramIO ℝ ℝ (FastRopeEmbedding.LogicalIO.Inputs r)
      (Fin r.heads → Fin (2*r.half) → ℝ) where
  core := Interface.programIO ideal r back row group
  inputs := FastRopeEmbedding.LogicalIO.inputEquiv r
  outputs := FastRopeEmbedding.LogicalIO.outputEquiv r

/-- The same memory channels interpreted by the retained native word graph. -/
def nativeIO (cfg : FP.Config) (r : Request) (back : Bool) (row group : Nat) :
    ProgramIO Numerics.Tagged Numerics.Tagged :=
  Interface.programIO (Numerics.operations (Numerics.softwarePrimitives cfg)) r back row group

end
end VeriTile.Bench.Repairs.RopeEmbedding
