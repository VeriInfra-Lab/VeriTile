import bench.repairs.tritonbench_g.cross_entropy2.Proofs.WholeGraph
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ReturnBacking
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Scheduled
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NegativeStorage

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical.Scheduled
open VeriTile.Triton Source Host Floating
open Floating.Scheduled (DeviceSchedule decode)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations FloatArray IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Backing EncodesNegation bytes byteOf flipSign)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
local notation "scheduledAlgebra" => Floating.Scheduled.algebra
local notation "scheduledHost" => Floating.Scheduled.hostArithmetic
set_option maxHeartbeats 8000000

/-- Every byte of every logical element of the actual returned object. The
native encoding and reference instruction ports are distinct parameters. -/
def ArrayBytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (state : World Compiled σ) (tensor : Tensor) (shape : List Nat) (expressions : Nat → Compiled) : Prop :=
  tensor.shape=shape ∧ tensor.dtype="torch.float32" ∧
  ∀ i, i<tensor.numel → ∀ byte : Fin 4,
    bytes layout (decode native profile) state (layout.buffer state.native tensor.region).device
      ((layout.buffer state.native tensor.region).base+4*tensor.index i+byte.val)=
    byteOf (if tensor.isNeg then flipSign (decode reference profile (expressions i))
      else decode reference profile (expressions i)) byte

theorem array_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode native profile) obs)
    (state : World Compiled σ) (tensor : Tensor) (shape : List Nat) (expressions : Nat → Compiled)
    (array : FloatArray obs state tensor shape expressions) (backed : Backing layout state.native tensor 4) :
    ArrayBytes layout native reference profile state tensor shape expressions := by
  refine ⟨array.shapeEq,array.dtype,?_⟩
  intro i hi byte
  have result := VeriTile.Bench.Repairs.CrossEntropy1.Physical.logical_value_bytes
    layout (decode native profile) obs negation state tensor i byte (expressions i)
      backed.active (backed.coordinates i hi) (array.values i hi)
  simpa only [decode,← ports_equal native reference ports] using result

theorem forward_graph_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Compiled σ) (negation : EncodesNegation (decode native profile) obs)
    (rt : Runtime Compiled σ) (lib : Library obs scheduledHost rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (schedule : DeviceSchedule) (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (success : Host.forward (scheduledAlgebra schedule o.logitScale) rt o x y initial=.ok returned final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region) (heldZ : kept returned.2.2.region) :
    ∃ (e : ForwardData (scheduledAlgebra schedule o.logitScale) rt o x y rows cols initial returned final) (k : ForwardKernelData obs e),
      ArrayBytes layout native reference profile final returned.1 [rows] (finalLoss k scheduledHost) ∧
      ArrayBytes layout native reference profile final returned.2.1 [rows] (finalLSE k scheduledHost) ∧
      ArrayBytes layout native reference profile final returned.2.2 [rows] (finalZ k scheduledHost) := by
  obtain ⟨e⟩ := forward_success (scheduledAlgebra schedule o.logitScale) rt o x y rows cols initial returned final shape success
  obtain ⟨k⟩ := forward_kernel_data obs scheduledHost rt lib e xLive yLive
  obtain ⟨loss,lse,z,rest⟩ := forward_values scheduledHost lib ints k yLive
  have backed := forward_return_backing life physical k loss.dtype lse.dtype z.dtype heldLoss heldLSE heldZ
  exact ⟨e,k,array_bytes layout native reference profile ports obs negation _ _ _ _ loss backed.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ lse backed.2.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ z backed.2.2⟩

/-- Standalone backward uses the caller-supplied LSE exactly. It is not
silently replaced by an independently recomputed normalizer. -/
theorem backward_graph_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Compiled σ) (negation : EncodesNegation (decode native profile) obs)
    (rt : Runtime Compiled σ) (lib : Library obs scheduledHost rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (schedule : DeviceSchedule) (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (success : Host.backward (scheduledAlgebra schedule o.logitScale) rt o g x l y initial=.ok output final) (held : kept output.region) :
    ArrayBytes layout native reference profile final output [rows,cols]
      (fun i => gradientScalar (scheduledAlgebra schedule o.logitScale) o (obs.integer initial y (i/cols))
        (value obs initial x i) (value obs initial l (i/cols)) (value obs initial g (i/cols)) (i%cols)) := by
  obtain ⟨e⟩ := backward_success (scheduledAlgebra schedule o.logitScale) rt o g x l y rows cols initial output final shape success
  obtain ⟨k⟩ := backward_kernel_data obs scheduledHost rt lib e shape xLive yLive gLive lLive
  have array := backward_array k shape
  have backed := backward_return_backing life physical k held
  apply array_bytes layout native reference profile ports obs negation _ _ _ _ _ backed
  refine ⟨array.shapeEq,array.dtype,array.live,?_⟩
  intro i hi
  have size : output.numel=rows*cols := by simp [Tensor.numel,array.shapeEq]
  have slot := VeriTile.Bench.Repairs.CrossEntropy1.Host.flattened_slot rows cols i (by simpa only [size] using hi)
  have graph := backward_graph_lane scheduledHost lib k shape yLive gLive lLive (i/cols) (i%cols) slot.2.1 slot.1
  rw [slot.2.2] at graph
  exact (array.values i hi).trans graph

/-- Complete returned-byte composition for the explicit source/host graph.
The saved LSE, allocations and output identities are derived inside the
proof. The aligned schedule carries explicit FMA/select instructions all
the way into physical readback. Scalar conformance remains local. -/
theorem complete_returned_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Compiled σ) (negation : EncodesNegation (decode native profile) obs)
    (rt : Runtime Compiled σ) (lib : Library obs scheduledHost rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (schedule : DeviceSchedule) {o : Options} {g x y : Tensor} {rows cols : Nat} {initial saved final : World Compiled σ}
    {returned : Return} {output : Tensor}
    {e : ForwardData (scheduledAlgebra schedule o.logitScale) rt o x y rows cols initial returned saved} (k : ForwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region)
    (b : BackwardData (scheduledAlgebra schedule o.logitScale) rt o g x returned.2.1 y rows cols saved output final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    ArrayBytes layout native reference profile saved returned.1 [rows] (finalLoss k scheduledHost) ∧
    ArrayBytes layout native reference profile saved returned.2.1 [rows] (finalLSE k scheduledHost) ∧
    ArrayBytes layout native reference profile saved returned.2.2 [rows] (finalZ k scheduledHost) ∧
    ArrayBytes layout native reference profile final output [rows,cols]
      (fun i => gradientScalar (scheduledAlgebra schedule o.logitScale) o (obs.integer initial y (i/cols))
        (value obs initial x i) (finalLSE k scheduledHost (i/cols)) (value obs saved g (i/cols)) (i%cols)) := by
  obtain ⟨loss,lse,z,gradient⟩ := complete_graph obs (scheduledAlgebra schedule o.logitScale) scheduledHost rt lib ints k shape xLive yLive gLive b
  have forwardBacking := forward_return_backing life physical k loss.dtype lse.dtype z.dtype heldLoss heldLSE heldZ
  have forwardFrame := (forward_values scheduledHost lib ints k yLive).2.2.2.2.2.2.2
  obtain ⟨bk⟩ := backward_kernel_data obs scheduledHost rt lib b shape
    (forwardFrame.live _ xLive) (forwardFrame.live _ yLive) gLive lse.live
  have backwardBacking := backward_return_backing life physical bk heldOutput
  exact ⟨array_bytes layout native reference profile ports obs negation _ _ _ _ loss forwardBacking.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ lse forwardBacking.2.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ z forwardBacking.2.2,
    array_bytes layout native reference profile ports obs negation _ _ _ _ gradient backwardBacking⟩

#axiomsClean array_bytes
#axiomsClean forward_graph_bytes
#axiomsClean backward_graph_bytes
#axiomsClean complete_returned_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical.Scheduled
