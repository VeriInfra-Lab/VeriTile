import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.PublicSource
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Contract
namespace VeriTile.Bench.Repairs.LayerNormTriton.Checked
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {caller : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}

/-- Ordinary queries on the same physical Tensor handles used by the host.
No value, result, native-safety or kernel-success observation occurs here. -/
structure Observations (lib : APIHost api neg caller Metadata) where
  tensor : Storage.Tensor → VeriTile.Torch.TensorMetadata
  shape : ∀ t,(tensor t).shape=t.view.shape
  dtype : ∀ t,(tensor t).dtype=.floating .fp32
  layout : ∀ t,(tensor t).layout=.strided
  device : ∀ t,(tensor t).device.index=t.view.device
  negative : ∀ t,(tensor t).isNeg=t.view.negative
  pointer : ∀ t,0<(tensor t).numel → (tensor t).dataPtr.toNat=t.view.allocation.base
  contiguous : ∀ t,lib.isContiguous t=(tensor t).isContiguous

def epsilon (value : Option Float) : Compiler.Shared.HostScalar :=
  .binary64 (value.getD 0.0).toBits


def forward (lib : APIHost api neg caller Metadata) (observations : Observations lib)
    (service : Host.ForwardService platform api neg caller Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (normalized : Option (List Nat)) (eps : Option Float)
    (initial : State Store Metadata) : Except Unit (Outcome (State Store Metadata) Dense) :=
  if Contract.inputCheck (observations.tensor x) (observations.tensor w) (observations.tensor b) normalized eps then
    .ok ((sourceForward lib service identity x w b (normalized.getD []) (epsilon eps)).run initial)
  else .error ()

theorem forward_admitted (lib : APIHost api neg caller Metadata) (observations : Observations lib)
    (service : Host.ForwardService platform api neg caller Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (normalized : Option (List Nat)) (eps : Option Float)
    (initial : State Store Metadata) (actual : Outcome (State Store Metadata) Dense)
    (h : forward lib observations service identity x w b normalized eps initial=.ok actual) :
    Contract.InputContract (observations.tensor x) (observations.tensor w) (observations.tensor b) normalized eps ∧
      actual=((sourceForward lib service identity x w b (normalized.getD []) (epsilon eps)).run initial) := by
  unfold forward at h
  split at h
  · exact ⟨Contract.inputCheck_iff _ _ _ _ _ |>.mp ‹_›,Except.ok.inj h.symm⟩
  · contradiction

def backward (lib : APIHost api neg caller Metadata) (observations : Observations lib)
    (extra : BackwardHost.APIExtra api neg caller Metadata)
    (service : BackwardHost.Service platform api neg caller Metadata)
    (identity : Nat) (c : SavedContext) (b : Dense) (saved : BackwardHost.Saved c b)
    (dy : Storage.Tensor) (initial : State Store Metadata)
    (lookup : lib.context initial.metadata identity=some c) :
    Except Unit (Outcome (State Store Metadata) (List (Option Storage.Tensor))) :=
  let accessed := (call .saveOriginal (lib.accessSaved identity).call).run initial
  match accessed.result with
  | .error e => .ok ⟨accessed.state,.error e,accessed.trace⟩
  | .ok _ =>
    if Contract.gradientCheck (observations.tensor dy) (observations.tensor c.x.tensor) c.inputShape then
      .ok ((inDevice (lib.enter c.x.tensor.view.device).call (lib.leave accessed.state.owner).call
        (BackwardHost.body lib extra service c b saved dy)).run accessed.state)
    else .error ()

theorem backward_admitted (lib : APIHost api neg caller Metadata) (observations : Observations lib)
    (extra : BackwardHost.APIExtra api neg caller Metadata)
    (service : BackwardHost.Service platform api neg caller Metadata)
    (identity : Nat) (c : SavedContext) (b : Dense) (saved : BackwardHost.Saved c b)
    (dy : Storage.Tensor) (initial : State Store Metadata)
    (lookup : lib.context initial.metadata identity=some c)
    (actual : Outcome (State Store Metadata) (List (Option Storage.Tensor))) (out : List (Option Storage.Tensor))
    (h : backward lib observations extra service identity c b saved dy initial lookup=.ok actual)
    (success : actual.result=.ok out) :
    Contract.GradientContract (observations.tensor dy) (observations.tensor c.x.tensor) c.inputShape ∧
      actual=(sourceBackward lib extra service identity c b saved dy initial lookup).outcome := by
  unfold backward at h
  cases hx : ((call .saveOriginal (lib.accessSaved identity).call).run initial).result with
  | error e =>
    simp only [hx] at h
    cases h
    contradiction
  | ok value =>
    simp only [hx] at h
    split at h
    · refine ⟨Contract.gradientCheck_iff _ _ _ |>.mp ‹_›,?_⟩
      cases h
      simp only [sourceBackward,BackwardHost.backward]
      split <;> simp_all
    · contradiction

#axiomsClean forward_admitted
#axiomsClean backward_admitted
end
end VeriTile.Bench.Repairs.LayerNormTriton.Checked
