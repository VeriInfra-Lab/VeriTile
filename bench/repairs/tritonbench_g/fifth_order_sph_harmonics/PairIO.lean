import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.IO
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOPair
open VeriTile Triton FP.Structural HarmonicIO KernelScalar
noncomputable section

def backend : Grid.Direction → Algebra Packing.Word
  | .forward => forwardBackend selectedProfile
  | .backward => backwardBackend selectedProfile

def io (d : Grid.Direction) (p : Parameters) : ProgramIO Packing.Word Packing.Word :=
  match d with
  | .forward => forwardIO (backend d) p
  | .backward => backwardIO (backend d) p

def job (M : Algebra Packing.Word) (d : Grid.Direction) (p : Parameters) (pid : Nat) : Invocation Packing.Word where
  kernel := SourcePair.original d p
  backend := M
  pids := fun _ => pid
  setPids := fun ax => ax == 0

def candidateJob (M : Algebra Packing.Word) (d : Grid.Direction) (p : Parameters) (pid : Nat) : Invocation Packing.Word :=
  { job M d p pid with kernel := SourcePair.candidate d p }

def schedule (d : Grid.Direction) (p : Parameters) : List (Invocation Packing.Word) :=
  (List.range (Grid.blocks p)).map (candidateJob (backend d) d p)

def variant (d : Grid.Direction) (p : Parameters) : ProgramIO.Variant (io d p) where
  schedule := fun _ => schedule d p
  observe := (io d p).observe

def target (d : Grid.Direction) (p : Parameters) : (io d p).Inputs → (io d p).Outputs :=
  match d with
  | .forward => forwardTarget (backend .forward) p nativeForward
  | .backward => backwardTarget (backend .backward) p nativeBackward

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOPair
