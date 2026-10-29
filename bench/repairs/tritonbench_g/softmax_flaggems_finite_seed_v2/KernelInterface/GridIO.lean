import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPGrid
import VeriTile.Triton.Float.ProgramIO

/-! Concrete whole-grid invocations of the unchanged source. The public schedule
installs exactly the existing grid metadata and clears only local registers. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

def job (A : Algebra α) (source : ComputeKernel) (gx gy : Nat) (p : Program) : Invocation α where
  kernel := source
  backend := A
  clearRegisters := Bool.true
  pids := fun a => if a=0 then p.1 else if a=1 then p.2 else 0
  numPids := some (fun a => if a=0 then gx else if a=1 then gy else 1)

def forwardJobs (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) : List (Invocation α) :=
  let g := FPGrid.forwardGrid d c
  (grid g.1 g.2).map (job A (FPGrid.forwardKernel X Y d c) g.1 g.2)

def backwardJobs (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) : List (Invocation α) :=
  let g := FPGrid.backwardGrid d c
  (grid g.1 g.2).map (job A (FPGrid.backwardKernel P G Y d c) g.1 g.2)

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
