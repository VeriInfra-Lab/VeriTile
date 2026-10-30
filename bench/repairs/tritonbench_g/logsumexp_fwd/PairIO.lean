import bench.repairs.tritonbench_g.logsumexp_fwd.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LogsumexpFwd.PairIO
open VeriTile.Triton FP.Structural Interface
noncomputable section
set_option maxHeartbeats 3000000
variable {A : Type} [Inhabited A]

def candidateInvocation (engine : Engine A) (a : RawKernel.Geometry) (scale : A) (j : Nat) : Invocation A :=
  { invocation engine a scale j with kernel := rowBlockCandidate "input" "scratch" a.columns a.block a.hasScale }

def variant (engines : Nat → Engine A) (a : RawKernel.Geometry) (scale : A) (jobs : List Nat) :
    ProgramIO.Variant (programIO engines a scale jobs) where
  schedule := fun _ => jobs.map (fun j => candidateInvocation (engines j) a scale j)
  observe := (programIO engines a scale jobs).observe

def original (engines : Nat → Engine A) (a : RawKernel.Geometry) (scale : A) (jobs : List Nat) :=
  ProgramIO.Implementation.original (programIO engines a scale jobs) (fun _ => True)

def candidate (engines : Nat → Engine A) (a : RawKernel.Geometry) (scale : A) (jobs : List Nat) :=
  ProgramIO.Implementation.ofVariant (variant engines a scale jobs) (fun _ => True)

end
end VeriTile.Bench.Repairs.LogsumexpFwd.PairIO
