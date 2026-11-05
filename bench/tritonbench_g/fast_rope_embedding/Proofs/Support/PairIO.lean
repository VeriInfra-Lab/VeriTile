import bench.tritonbench_g.fast_rope_embedding.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.ProgramPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

structure Program where
  kernel : ComputeKernel
  launches : List (Nat × Nat)

def job (M : Algebra α) (kernel : ComputeKernel) (launch : Nat × Nat) : Invocation α where
  kernel := kernel
  backend := M
  pids := fun ax => if ax = 0 then launch.1 else if ax = 1 then launch.2 else 0

def schedule (M : Algebra α) (p : Program) : List (Invocation α) := p.launches.map (job M p.kernel)

def grid (r : Request) : List (Nat × Nat) :=
  if r.half = 0 then [] else (List.range r.rows).flatMap
    (fun row => (List.range r.groups).map (fun group => (row,group)))

def original (r : Request) (qStride cosineStride sineStride : Nat) (backward : Bool)
    (launches : List (Nat × Nat)) : Program :=
  ⟨Kernels.rope "Q" "C" "S" qStride cosineStride sineStride
    r.sequence r.dimension r.heads r.block backward, launches⟩
def candidate (r : Request) (qStride cosineStride sineStride : Nat) (backward : Bool)
    (launches : List (Nat × Nat)) : Program :=
  ⟨Kernels.ropeIndexCommuted "Q" "C" "S" qStride cosineStride sineStride
    r.sequence r.dimension r.heads r.block backward, launches⟩



def candidateIO {α : Type} (op : Operations α) (r : Request)
    (QR CR SR : Nat) (backward : Bool) (row group : Nat) : ProgramIO α α :=
  { Interface.programIO op r QR CR SR backward row group with
    schedule := fun _ => schedule op.backend (candidate r QR CR SR backward [(row,group)]) }

end
end VeriTile.Bench.Repairs.FastRopeEmbedding.ProgramPair

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.IOPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

def io (cfg : FP.Config) (r : Request) (QR CR SR : Nat) (backward : Bool) (row group : Nat) :=
  Interface.programIO (Numerics.operations (Numerics.softwarePrimitives cfg)) r QR CR SR backward row group

def variant (cfg : FP.Config) (r : Request) (QR CR SR : Nat) (backward : Bool) (row group : Nat) :
    ProgramIO.Variant (io cfg r QR CR SR backward row group) where
  schedule := (ProgramPair.candidateIO (Numerics.operations (Numerics.softwarePrimitives cfg))
    r QR CR SR backward row group).schedule
  observe := (io cfg r QR CR SR backward row group).observe

end
end VeriTile.Bench.Repairs.FastRopeEmbedding.IOPair
