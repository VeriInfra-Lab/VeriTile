import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelContract
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelBinding
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Contract
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Host

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Checked
open VeriTile.Triton Host
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Step bind)
open CrossEntropy1.TensorModel (Tensor)
noncomputable section

abbrev ForwardMetadata := CrossEntropy1.Checked.ForwardMetadata
abbrev BackwardMetadata := CrossEntropy1.Checked.BackwardMetadata

def InputContract (m : ForwardMetadata) (scaleFloat scaleFinite : Bool)
    (o : Options) (logits labels : Tensor) (target : Nat → Int) : Prop :=
  CrossEntropy1.Checked.InputContract m o.toOptions logits labels target ∧
  scaleFloat=Bool.true ∧ scaleFinite=Bool.true

abbrev BackwardContract := CrossEntropy1.Checked.BackwardContract

def forward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (m : ForwardMetadata) (scaleFloat scaleFinite : Bool) (o : Options)
    (logits labels : Tensor) (target : Nat → Int) (initial : World α σ) :
    Step (World α σ) ForwardReturn := by
  classical
  exact if InputContract m scaleFloat scaleFinite o logits labels target then Host.forward M rt o logits labels initial
    else .error "checked forward contract" initial

/-- The guard wrapper retrieves saved tensors before inspecting the upstream;
the preserved resident backward body then performs its unchanged retrieval. -/
def backward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (c : Context) (saved : σ) (m : BackwardMetadata)
    (rows : Nat) (upstream : Tensor) (zUpstream : Option Tensor) (initial : World α σ) :
    Step (World α σ) Tensor := by
  classical
  exact bind (CrossEntropy1.Autograd.retrieve versions (CrossEntropy1.Autograd.capture versions c.base saved) initial)
    fun _ ready => if BackwardContract m rows c.logits upstream then
      Host.backwardEntry M rt versions c saved upstream zUpstream ready
    else .error "checked backward contract" ready

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Checked
