import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Contract
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SourceBindings
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulBackward
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Addresses

namespace VeriTile.Bench.Repairs.FastLayernorm
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {caller : List Storage.Tensor} {lib : Host.APIHost api neg caller Metadata}

def originalSHA256 : String := Compiler.originalSHA256
def candidateSHA256 : String := Compiler.sourceSHA256
abbrev forwardProgram := @Runtime.kernelAction
abbrev backwardProgram := @BackwardRuntime.kernelAction
abbrev sourceForward := @ForwardCall.publicForward
abbrev sourceBackward := @BackwardCall.invoke

/-- Ordinary metadata/scalar queries on actual handles, including the actual
data pointer and the contiguous() identity predicate. No output law occurs. -/
structure Observations (lib : Host.APIHost api neg caller Metadata) where
  tensor : Storage.Tensor → VeriTile.Torch.TensorMetadata
  scalar : Compiler.Shared.HostScalar → Contract.Epsilon
  shape : ∀ t,(tensor t).shape=t.view.shape
  dtype : ∀ t,(tensor t).dtype=.floating .fp32
  layout : ∀ t,(tensor t).layout=.strided
  device : ∀ t,(tensor t).device.index=t.view.device
  negative : ∀ t,(tensor t).isNeg=t.view.negative
  pointer : ∀ t,0<(tensor t).numel → (tensor t).dataPtr.toNat=t.view.allocation.base
  contiguous : ∀ t,lib.isContiguous t=(tensor t).isContiguous
  viewPointer : ∀ t shape s u,((lib.view t shape).call.run s).2=.ok u →
    u.tensor.view.allocation.base=t.tensor.view.allocation.base
  viewOffset : ∀ t shape s u,((lib.view t shape).call.run s).2=.ok u →
    u.tensor.storageOffset=t.tensor.storageOffset

structure Domain (observations : Observations lib) (module : Wrapper.Module) (x : Storage.Tensor)
    (caller : List Storage.Tensor) (initial : Host.State Store Metadata) : Prop where
  admitted : Contract.InputContract (observations.tensor x) (observations.tensor module.weight)
    (module.bias.map observations.tensor) (observations.scalar module.selectedEpsilon)
  affine : module.affine=Bool.true
  xLive : initial.live x
  weightLive : initial.live module.weight
  biasLive : ∀ b,b∈module.bias.toList → initial.live b
  xCaller : x∈caller
  weightCaller : module.weight∈caller
  biasCaller : ∀ b,b∈module.bias.toList → b∈caller

def GradientDomain (observations : Observations lib) (c : Host.SavedContext) (dy : Storage.Tensor) : Prop :=
  Contract.GradientContract (observations.tensor dy)
    {observations.tensor c.x.tensor with shape:=c.inputShape}

/-- The forward contiguous calls are proved identities from the same checked
metadata. Ordinary view calls retain actual backing and storage offset. -/
theorem materialize_same_storage (observations : Observations lib)
    (r : Preparation.MaterializeRoute lib owner t shape initial)
    (contiguous : (observations.tensor t).isContiguous=Bool.true) :
    r.output.tensor.backing=t.backing := by
  have identity := Host.dense_identity r.dense ((observations.contiguous t).trans contiguous)
  exact (lib.viewed r.dense.output shape r.dense.final r.output r.viewOK).backing.trans
    (congrArg (fun t : Storage.Tensor => t.backing) identity)

theorem forward_input_storage (observations : Observations lib)
    (domain : Domain observations module x caller initial)
    (f : ForwardCall.SuccessfulCall lib metadata service identity x module.weight module.bias module.selectedEpsilon initial) :
    f.body.preparation.data.x.tensor.backing=x.backing ∧
    f.body.preparation.data.w.tensor.backing=module.weight.backing :=
  ⟨materialize_same_storage observations f.body.preparation.xr (domain.admitted.input.dense rfl),
    materialize_same_storage observations f.body.preparation.wr (domain.admitted.weight.dense rfl)⟩

/-- Ordinary views may change shape/strides but preserve the original storage,
data pointer and storage offset, including empty and singleton dimensions. -/
structure SameStorage (u t : Storage.Tensor) : Prop where
  backing : u.backing=t.backing
  pointer : u.view.allocation.base=t.view.allocation.base
  offset : u.storageOffset=t.storageOffset

theorem materialize_same_view (observations : Observations lib)
    (r : Preparation.MaterializeRoute lib owner t shape initial)
    (contiguous : (observations.tensor t).isContiguous=Bool.true) :
    SameStorage r.output.tensor t := by
  have identity := Host.dense_identity r.dense ((observations.contiguous t).trans contiguous)
  refine ⟨materialize_same_storage observations r contiguous,?_,?_⟩
  · exact (observations.viewPointer _ _ _ _ r.viewOK).trans
      (congrArg (fun t : Storage.Tensor => t.view.allocation.base) identity)
  · exact (observations.viewOffset _ _ _ _ r.viewOK).trans
      (congrArg (fun t : Storage.Tensor => t.storageOffset) identity)

theorem optional_same_view (observations : Observations lib)
    (r : Preparation.OptionalRoute lib owner initial bias)
    (contiguous : ∀ b,bias=some b → (observations.tensor b).isContiguous=Bool.true)
    (original : Storage.Tensor) (prepared : Storage.Dense)
    (ho : bias=some original) (hp : r.output=some prepared) :
    SameStorage prepared.tensor original := by
  cases r with
  | none => contradiction
  | @some b route =>
    have he : prepared=route.output := Option.some.inj hp.symm
    subst prepared
    have he : b=original := Option.some.inj ho
    subst original
    exact materialize_same_view observations route (contiguous b rfl)

theorem forward_input_views (observations : Observations lib)
    (domain : Domain observations module x caller initial)
    (f : ForwardCall.SuccessfulCall lib metadata service identity x module.weight module.bias module.selectedEpsilon initial) :
    SameStorage f.body.preparation.data.x.tensor x ∧
    SameStorage f.body.preparation.data.w.tensor module.weight ∧
    (∀ original prepared,module.bias=some original → f.body.preparation.data.bias=some prepared →
      SameStorage prepared.tensor original) := by
  refine ⟨materialize_same_view observations f.body.preparation.xr (domain.admitted.input.dense rfl),
    materialize_same_view observations f.body.preparation.wr (domain.admitted.weight.dense rfl),?_⟩
  exact optional_same_view observations f.body.preparation.br
    (fun b hb => (domain.admitted.biasTensor _ (by simp [hb])).dense rfl)

theorem forward_original_frame (observations : Observations lib)
    (domain : Domain observations module x caller initial)
    (t : Storage.Tensor) (member : t=x ∨ t=module.weight ∨ t∈module.bias.toList)
    (offset : Nat) (bound : offset<t.backing.nbytes) :
    Storage.observe api ((sourceForward lib metadata service identity module x).run initial).state.memory
      (t.backing.owner,t.backing.base+offset)=
      Storage.observe api initial.memory (t.backing.owner,t.backing.base+offset) := by
  apply ForwardCall.public_frame
  refine ⟨t,?_,rfl,by omega,by omega⟩
  rcases member with hx|hw|hb
  · subst t; exact domain.xCaller
  · subst t; exact domain.weightCaller
  · exact domain.biasCaller t hb

theorem forward_output_device
    (f : ForwardCall.SuccessfulCall lib metadata service identity x w bias eps initial) :
    f.body.output.tensor.view.device=x.view.device := by
  have hs := (ForwardCall.output_storage f).1
  have hy := (lib.emptyLikeResult f.body.preparation.xr.output f.body.preparation.br.final
    f.body.preparation.y f.body.preparation.yOK).2.2
  exact f.body.output.tensor.owner.trans
    ((congrArg (fun b : Storage.Backing => b.owner) hs).trans
      (hy.trans (Preparation.materialize_metadata f.body.preparation.xr).2.2.1))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm
