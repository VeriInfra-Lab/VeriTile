import bench.tritonbench_g.chunk_gate_recurrence.IO
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.ProgramPair

/-! Two complete source programs on the unchanged public kernel IO. Both use
the same raw-word instruction ports and complete DG reduction plan. Only the
source addition order and its product-left/product-right FMA recognition differ.
The chosen kernel boundary is one legal 64×16 program with its entire time loop,
including every output family and the full exterior-cell frame. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
open scoped VeriTile.Spec
noncomputable section

/-- The original declared model includes its recorded N=1 backward operand order. -/
def originalIO (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) : ProgramIO Node Node :=
  IO.program (model isa plan kind c) kind c s seed p

/-- The existing sum-left candidate has identical layout, encoding, PID binding,
input/output windows and observations. Its actual quotation is in Kernels. -/
def sumLeftIO (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    ProgramIO.Variant (originalIO isa plan kind c s seed p) where
  schedule := fun _ =>
    [{ IO.launch (ProgramPair.candidateAlgebra isa plan (swapped kind c)) kind c s seed p with
        kernel := ProgramPair.candidateSource kind c s seed }]
  observe := (originalIO isa plan kind c s seed p).observe

def original (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :=
  ProgramIO.Implementation.original (originalIO isa plan kind c s seed p) (fun _ => True)

def sumLeft (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :=
  ProgramIO.Implementation.ofVariant (sumLeftIO isa plan kind c s seed p) (fun _ => True)

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.PairIO
