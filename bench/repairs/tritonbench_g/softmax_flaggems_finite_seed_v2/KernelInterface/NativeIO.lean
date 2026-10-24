import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.IO
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Kernel

/-! Whole-tensor native observation with fixed stored FP32 input words. Initial
output words are an explicit read port, so aliased inputs and the exact sequential
grid remain admitted. Addressed input binding enforces coherence of aliased ports.
Each output keeps its own consumer-indexed reduction and lowering schedule. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural Native
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

def forwardNativeIO (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    AddressedProgramIO Word Term where
  toMemoryIO := layout Word [X,Y] Y (count d)
  schedule := fun _ => forwardJobs (algebra R) X Y d c
  encodeAt := fun r k w _ => .leaf r k w
  observe := Fin.cases
    (fun j t => Scalar.eval (Scalar.instructions ops scalar) (S (Y,j.1.val)) (t (Y,j.1.val)))
    (fun i => i.elim0)

def backwardNativeIO (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    AddressedProgramIO Word Term where
  toMemoryIO := layout Word [P,G,Y] Y (count d)
  schedule := fun _ => backwardJobs (algebra R) P G Y d c
  encodeAt := fun r k w _ => .leaf r k w
  observe := Fin.cases
    (fun j t => Scalar.eval (Scalar.instructions ops scalar) (S (Y,j.1.val)) (t (Y,j.1.val)))
    (fun i => i.elim0)

/-- A total reference extension of the finite stored-word ports. The graph uses
only valid tensor coordinates; the extension adds no backing-storage premise. -/
def forwardWords (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardNativeIO ops scalar R S X Y d c).Inputs) (r : RegionName) (k : Nat) : Word :=
  if h : k<count d then
    if r=X then v (0 : Fin 2) (⟨k,h⟩,PUnit.unit)
    else if r=Y then v (1 : Fin 2) (⟨k,h⟩,PUnit.unit) else 0
  else 0

def backwardWords (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (backwardNativeIO ops scalar R S P G Y d c).Inputs) (r : RegionName) (k : Nat) : Word :=
  if h : k<count d then
    if r=P then v (0 : Fin 3) (⟨k,h⟩,PUnit.unit)
    else if r=G then v (1 : Fin 3) (⟨k,h⟩,PUnit.unit)
    else if r=Y then v (2 : Fin 3) (⟨k,h⟩,PUnit.unit) else 0
  else 0

/-- Independent source-order native value graph, observed at the actual output
consumer. Primitive operations and schedules are explicit model parameters;
there is no native=Real claim or numerical transformation assumption. -/
def forwardNativeFormula (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardNativeIO ops scalar R S X Y d c).Inputs) :
    (forwardNativeIO ops scalar R S X Y d c).Outputs :=
  Fin.cases (fun j => Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,j.1.val)
    (((Kernel.forwardTarget R X Y d c (forwardWords ops scalar R S X Y d c v)).mem Y j.1.val).read .real)))
    (fun i => i.elim0)

def backwardNativeFormula (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (backwardNativeIO ops scalar R S P G Y d c).Inputs) :
    (backwardNativeIO ops scalar R S P G Y d c).Outputs :=
  Fin.cases (fun j => Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,j.1.val)
    (((Kernel.backwardTarget R P G Y d c (backwardWords ops scalar R S P G Y d c v)).mem Y j.1.val).read .real)))
    (fun i => i.elim0)

#specNonCircular forwardNativeFormula avoiding [SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun,FP.Structural.exec]
#specNonCircular backwardNativeFormula avoiding [SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
