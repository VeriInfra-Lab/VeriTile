import bench.tritonbench_g.context_attn_bloom.Proofs.HostInterface

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Host
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
set_option maxHeartbeats 1000000

def flatSource (a : Arguments) (r : Regions) (A : FlatAlloc) (swapped : Bool) : Kernel :=
  A.flattenKernel (source a r swapped).eraseDType.toAlgKernel

def active (a : Arguments) (r : Regions) (s : BlockState)
    (idx : GridIndex (grid a)) (i : RealIO.Lane (layout a)) : Prop :=
  RealIO.active (layout a) (launchOf a idx) (inputs s r) i

def address (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (idx : GridIndex (grid a)) (i : RealIO.Lane (layout a)) : MemCellAddr :=
  (A.flat, A.addr r.out (RealIO.write (layout a) (launchOf a idx) (inputs s r) i))

/-- Independent guarded normalized-block recurrence over the original tensor views. -/
def result (a : Arguments) (r : Regions) (s : BlockState)
    (idx : GridIndex (grid a)) (i : RealIO.Lane (layout a)) : ℝ :=
  RealIO.expected (layout a) (scale a) (launchOf a idx) (inputs s r) i

def writes (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState) : WriteFootprint :=
  HostComposition.allWrites (grid a) (active a r s) (address a r A s)

/-- Python updates the supplied O allocation and has no returned tensor. -/
def returned (final : BlockState) : BlockState × Option Unit := (final, none)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Host
