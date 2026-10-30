import bench.tritonbench_g.block_sparse_attn.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Native source selection and candidate invocations share PublicIO geometry. -/
namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairConnection
open VeriTile Triton FP.Structural
noncomputable section

def selectedCandidate (p : Parameters) := (quotientRemainder p).surfaceBody.take 27 ++
  [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
    (.ref .int [] "end_l") (.constNat 1) (compiledBody p)] ++
  (quotientRemainder p).surfaceBody.drop 28

def kernel (p : Parameters) : ComputeKernel :=
  .mk (quotientRemainder p).inputs (quotientRemainder p).outputs (selectedCandidate p)

end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairConnection

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairIO
open VeriTile Triton FP.Structural
noncomputable section

def invocation (ops : PhysicalKernel.NativePrimitives) (p : Parameters)
    (pid : PhysicalKernel.Program) : Invocation (Replicas Word) :=
  { PublicIO.invocation ops.arithmetic Bool.true p pid with kernel := PairConnection.kernel p }

def candidate (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (batches visits : Nat) :
    ProgramIO.Variant (PublicIO.nativeIO ops p batches visits) where
  schedule := fun _ => (PhysicalKernel.programs p batches).map (invocation ops p)
  observe := (PublicIO.nativeIO ops p batches visits).observe

end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairIO
