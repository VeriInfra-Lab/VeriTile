import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.StorageOrigins

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton
open CrossEntropy1.Host (World Step Observations TensorLibrary Groups IntegerViews CollectiveLibrary)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Integer reads use the ordinary tensor view. Floating sign handling is
fixed to the involutive marker operation, never supplied by a task caller. -/
def markerObservations {σ : Type} (integer : World Marker σ → Tensor → Nat → Int) : Observations Marker σ :=
  ⟨Marker.neg,integer⟩

/-- No old token may appear in a newly live allocation except at a coordinate
of the actual initialized FP32 result. The library rules below fix those
coordinates to the copied input or arithmetic result. This includes holes,
padding and real channels of newly allocated integer/bool tensors. -/
def FreshRemainder {σ : Type} (before after : World Marker σ) (output : Tensor) : Prop :=
  ∀ r i, after.live r → ¬before.live r →
    (r≠output.region ∨ output.dtype≠"torch.float32" ∨ ∀ j, j<output.numel → i≠output.index j) →
    (after.memory r i).read .real=Marker.computed

def FreshEmpty {σ : Type} (before after : World Marker σ) : Prop :=
  ∀ r i, after.live r → ¬before.live r → (after.memory r i).read .real=Marker.computed

/-- Ordinary allocation/initialization rules. `empty` never initializes a
token; copy/clone/resolve may only preserve the token at an actual copied
coordinate. No field mentions a kernel result or a caller-byte frame. -/
structure MarkerAllocationRules {σ : Type} (rt : Host.Runtime Marker σ) : Prop where
  empty : ∀ shape dtype device before output after,
    rt.empty shape dtype device before=.ok output after → FreshEmpty before after
  emptyLike : ∀ input packed before output after,
    rt.emptyLike input packed before=.ok output after → FreshEmpty before after
  resolve : ∀ input before output after,
    rt.resolveNeg input before=.ok output after → FreshRemainder before after output
  clone : ∀ input before output after,
    rt.cloneContiguous input before=.ok output after → FreshRemainder before after output
  contiguous : ∀ input before output after,
    rt.contiguous input before=.ok output after → FreshRemainder before after output
  logsumexp : ∀ input before output after,
    rt.logsumexp0 input before=.ok output after → FreshRemainder before after output
  sum : ∀ input before output after,
    rt.sum0 input before=.ok output after → FreshRemainder before after output
  square : ∀ input before output after,
    rt.square input before=.ok output after → FreshRemainder before after output
  scalarMul : ∀ scale input before output after,
    rt.scalarMul scale input before=.ok output after → FreshRemainder before after output
  equalInt : ∀ input target before output after,
    rt.equalInt input target before=.ok output after → FreshEmpty before after
  zerosLike : ∀ input before output after,
    rt.zerosLike input before=.ok output after → FreshRemainder before after output

/-- These laws describe only ordinary tensor primitives under the fixed
marker arithmetic. Fresh initialization supplements the existing reusable
copy/clone/resolve and arithmetic contracts. -/
structure MarkerPrimitiveRules {σ : Type} (integer : World Marker σ → Tensor → Nat → Int)
    (rt : Host.Runtime Marker σ) (groupSize groupRank : Nat → Nat) where
  library : Host.LocalAPI (markerObservations integer) rt
  arithmetic : TensorLibrary (markerObservations integer) markerHost rt.toRuntime
  zeros : Host.ZeroAPI (markerObservations integer) rt markerHost.zero
  allocation : MarkerAllocationRules rt
  integers : IntegerViews (markerObservations integer)
  collectives : CollectiveLibrary (markerObservations integer) markerHost rt.toRuntime groupSize groupRank

theorem empty_cannot_forge {σ : Type} {rt : Host.Runtime Marker σ} (rules : MarkerAllocationRules rt)
    (shape : List Nat) (dtype : String) (device : Nat) (before after : World Marker σ) (output : Tensor)
    (call : rt.empty shape dtype device before=.ok output after)
    (r : RegionName) (i : Nat) (live : after.live r) (fresh : ¬before.live r)
    (oldR : RegionName) (oldI : Nat) (sign : Bool) :
    (after.memory r i).read .real≠Marker.oldCell oldR oldI sign := by
  rw [rules.empty _ _ _ _ _ _ call r i live fresh]
  exact Marker.noConfusion

#axiomsClean empty_cannot_forge
#specNonCircular MarkerAllocationRules avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
