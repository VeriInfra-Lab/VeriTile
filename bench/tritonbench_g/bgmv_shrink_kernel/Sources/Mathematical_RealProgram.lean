import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealTail

/-! Full real execution of the actual signed source, including both output
branches and every split-K iteration. The formula reads only initial memory.
Allocation bounds and flat-memory transport are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealProgram
open VeriTile.Triton RealSetupSource RealSetup
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- This program's rank-slice GEMV contribution. Tail lanes outside K
contribute zero; K need not be divisible by the loop step. -/
def contribution (p : Parameters) (initial : BlockState) (slot n : Nat) : ℝ :=
  p.scaling * ∑ c ∈ Finset.range ((p.K + p.BK * p.splits - 1) / (p.BK * p.splits)),
    ∑ e : Fin p.BK,
      let k := c * (p.BK * p.splits) + (e.val + initial.pids 0 * p.BK)
      if k < p.K then
        initial.readMem p.input (initial.pids 1 * p.inputBatchStride + k) *
        initial.readMem p.weights (p.slotStride * slot + n * p.rowStride + k * p.colStride)
      else 0

/-- A single split overwrites output; multiple splits use atomic addition
and retain the cell's value from before this program. -/
def formula (p : Parameters) (initial : BlockState) (slot n : Nat) : ℝ :=
  if p.splits = 1 then contribution p initial slot n
  else initial.readMem p.output (RealTail.offset p (initial.pids 1) n) + contribution p initial slot n

def Misses (p : Parameters) (initial : BlockState) (r : RegionName) (o : Nat) : Prop :=
  ∀ n : Fin p.BN, n.val < p.N → ¬(p.output = r ∧ RealTail.offset p (initial.pids 1) n.val = o)

theorem contribution_eq (p : Parameters) (initial : BlockState) (slot n : Nat) :
    contribution p initial slot n = RealLoop.scaledContribution initial p.input p.weights slot p.scaling
      p.K p.inputBatchStride p.slotStride p.rowStride p.colStride p.BK p.splits n := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealProgram
