import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationTrace

namespace VeriTile.Bench.Repairs.CrossEntropy2.Invocation
open Source Floating Reduction
open Floating.Profiled
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
noncomputable section
set_option maxHeartbeats 10000000

def hasFMA : SmoothingProgram → Bool
  | .zero | .product _ | .scaled _ => Bool.false
  | .fused _ _ => Bool.true
  | .masked _ body => hasFMA body
  | .add a b => hasFMA a || hasFMA b

theorem pure_has_no_fma (tree : Tree) : hasFMA (pureSmoothing tree)=Bool.false := by
  induction tree with
  | leaf i | seed => rfl
  | node a b ia ib => simp only [pureSmoothing,hasFMA,ia,ib,Bool.false_or]

theorem smoothing_tree_fma (tree : Tree) (inputs : Nat → SmoothingProgram) :
    hasFMA (tree.run SmoothingProgram.add .zero inputs)=
      tree.run (fun a b => a || b) Bool.false (fun i => hasFMA (inputs i)) := by
  induction tree with
  | leaf i | seed => rfl
  | node a b ia ib => simp only [Tree.run,hasFMA,ia,ib]

theorem aligned_chain_has_fma (lane n : Nat) : hasFMA (alignedSmoothingChain lane n)=Bool.true := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [alignedSmoothingChain,hasFMA,ih,Bool.false_or]

theorem aligned_has_fma : hasFMA alignedPlan.smoothing=Bool.true := by
  change hasFMA alignedSmoothing=Bool.true
  rw [alignedSmoothing,smoothing_tree_fma]
  simp only [aligned_chain_has_fma]
  decide

theorem aligned_offset4_distinct : alignedPlan≠offset4Plan := by
  intro equal
  have h := congrArg (fun p : DevicePlan 65536 => hasFMA p.smoothing) equal
  change hasFMA alignedPlan.smoothing=hasFMA offset4Plan.smoothing at h
  rw [aligned_has_fma] at h
  have pure : hasFMA offset4Plan.smoothing=Bool.false := pure_has_no_fma sm90Large.tree
  rw [pure] at h
  contradiction

/-- A single total backend API simultaneously selects two different actual
observed graphs at width 65536. Other widths are structural witnesses only. -/
def dualDevice (key : Scheduled.JITKey) : DeviceFamily := fun width positive =>
  if equal : width=65536 then by
    subst width
    exact if key=Scheduled.alignedKey then alignedPlan else offset4Plan
  else witnessDevice width positive

def dualABI : BackendABI := {witnessABI with device:=dualDevice}

theorem dual_aligned : dualABI.device Scheduled.alignedKey 65536 (by decide)=alignedPlan := by
  simp [dualABI,dualDevice]

theorem dual_offset4 : dualABI.device Scheduled.offset4Key 65536 (by decide)=offset4Plan := by
  have different : Scheduled.offset4Key≠Scheduled.alignedKey := by decide
  simp [dualABI,dualDevice,different]

theorem same_width_different_graphs :
    dualABI.device Scheduled.alignedKey 65536 (by decide)≠
      dualABI.device Scheduled.offset4Key 65536 (by decide) := by
  rw [dual_aligned,dual_offset4]
  exact aligned_offset4_distinct

/-- No global profile-equality premise is needed to execute either call.
This applies twice to the same runtime with two successful setup operations. -/
theorem launch_uses_actual_key {σ : Type} (layout : Layout σ) (abi : BackendABI)
    (rt : Host.Runtime Compiled σ) (p : Parameters) (count : Nat) (before ready : World Compiled σ)
    (nonempty : p.rows≠0) (setup : rt.setupForward2 p count before=.ok () ready) :
    forwardLaunch layout abi rt p count before=
      .ok (abi.device (Scheduled.keyFor layout ready p))
        {ready with memory:=(Grid.forward
          (Floating.Profiled.algebra (abi.device (Scheduled.keyFor layout ready p)) p.logitScale) p ready.memory count)} := by
  simp only [forwardLaunch,if_neg nonempty,setup,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind]

#axiomsClean same_width_different_graphs
#axiomsClean launch_uses_actual_key
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Invocation
