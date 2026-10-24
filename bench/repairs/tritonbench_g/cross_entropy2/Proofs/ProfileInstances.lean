import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfileDispatch

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
open Reduction
noncomputable section

/-- A pure reduction tree keeps every rounded masked product and every seed.
This conversion introduces neither fusion nor reassociation. -/
def pureSmoothing : Tree → SmoothingProgram
  | .leaf i => .scaled i
  | .seed => .zero
  | .node a b => .add (pureSmoothing a) (pureSmoothing b)

theorem pure_smoothing_leaves (tree : Tree) : (pureSmoothing tree).leaves=tree.leaves := by
  induction tree with
  | leaf i | seed => rfl
  | node a b ia ib => simp only [pureSmoothing,SmoothingProgram.leaves,Tree.leaves,ia,ib]

theorem pure_smoothing_exact (n : Ports) (scale : Word) (input : Nat → Word)
    (active : Nat → Bool) (tree : Tree) :
    (pureSmoothing tree).run n scale input active=
      tree.run n.add (n.literal 0) (fun i => if active i then n.mul scale (input i) else n.literal 0) := by
  induction tree with
  | leaf i | seed => rfl
  | node a b ia ib => simp only [pureSmoothing,SmoothingProgram.run,Tree.run,ia,ib]

def purePlan {width : Nat} (maximum exponentialSum smoothing : Plan width) : DevicePlan width where
  maximum := maximum
  exponentialSum := exponentialSum
  smoothing := pureSmoothing smoothing.tree
  smoothingCovers := (pure_smoothing_leaves smoothing.tree) ▸ smoothing.covers

/-- The observed aligned SM90 instance retains its three FMAs and shared
mask. Its maximum and exponential sum remain seedless pure trees. -/
def alignedPlan : DevicePlan 65536 :=
  ⟨sm90AlignedLarge,sm90AlignedLarge,alignedSmoothing,aligned_smoothing_covers⟩

/-- The observed offset4 instance uses 512 scalar registers per thread,
then the exact warp/shared stages, for all three reductions. -/
def offset4Plan : DevicePlan 65536 := purePlan sm90Large sm90Large sm90Large

/-- A total structural witness, establishing that the interface is inhabited.
Unobserved widths in this witness are not installed-compiler claims. -/
def witnessDevice : DeviceFamily := fun width positive =>
  purePlan (witnessProfile.maxTree width positive) (witnessProfile.sumTree width positive)
    (witnessProfile.sumTree width positive)

/-- A total graph API witness, not a claim about an installed backend. -/
def witnessABI : BackendABI where
  device := fun _ => witnessDevice
  torchSum := fun _ => witnessProfile.hostSumTree
  torchLSEMax := fun _ => witnessProfile.hostMaxTree
  torchLSESum := fun _ => witnessProfile.hostLSESumTree

#axiomsClean pure_smoothing_leaves
#axiomsClean pure_smoothing_exact
#axiomsClean alignedPlan
#axiomsClean offset4Plan
#axiomsClean witnessABI
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
