import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.IO
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Candidate IO reuses the original grouped matrix interface. -/
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
variable {α : Type} [Inhabited α]

def launch (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) (pid : Nat) : Invocation α :=
  { IO.launch rest P p pid with kernel := maskHoisted p }
def candidateIO (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) : ProgramIO α α :=
  { IO.matmul rest P p with schedule := fun _ =>
      (List.range (Geometry.count (Core.geometry p))).map (launch rest P p) }

def variant (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :
    ProgramIO.Variant (IO.matmul rest P p) where
  schedule := (candidateIO rest P p).schedule
  observe := (IO.matmul rest P p).observe

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
