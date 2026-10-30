import bench.repairs.tritonbench_g.cross_entropy2.Proofs.WholeGraph
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ReturnBacking
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.CompilerRewrite
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NegativeStorage

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations FloatArray IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Backing EncodesNegation bytes byteOf flipSign)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Every byte of every logical element of the actual returned object. The
native encoding and reference instruction ports are distinct parameters. -/
def ArrayBytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (state : World Expr σ) (tensor : Tensor) (shape : List Nat) (expressions : Nat → Expr) : Prop :=
  tensor.shape=shape ∧ tensor.dtype="torch.float32" ∧
  ∀ i, i<tensor.numel → ∀ byte : Fin 4,
    bytes layout (word native profile) state (layout.buffer state.native tensor.region).device
      ((layout.buffer state.native tensor.region).base+4*tensor.index i+byte.val)=
    byteOf (if tensor.isNeg then flipSign (runInstruction reference profile (compile (expressions i)).value)
      else runInstruction reference profile (compile (expressions i)).value) byte

theorem array_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (obs : Observations Expr σ)
    (negation : EncodesNegation (word native profile) obs)
    (state : World Expr σ) (tensor : Tensor) (shape : List Nat) (expressions : Nat → Expr)
    (array : FloatArray obs state tensor shape expressions) (backed : Backing layout state.native tensor 4) :
    ArrayBytes layout native reference profile state tensor shape expressions := by
  refine ⟨array.shapeEq,array.dtype,?_⟩
  intro i hi byte
  have result := VeriTile.Bench.Repairs.CrossEntropy1.Physical.logical_value_bytes
    layout (word native profile) obs negation state tensor i byte (expressions i)
      backed.active (backed.coordinates i hi) (array.values i hi)
  simpa only [compiled_word,← ports_equal native reference ports] using result

/-- A checked sequence of local instruction rewrites transports each proved
returned byte to a concrete instruction graph. The certificate type has no
whole-expression equality constructor. -/
theorem certified_array_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (laws : Peepholes native) (one : ConstantOne native)
    (state : World Expr σ) (tensor : Tensor) (shape : List Nat) (expressions : Nat → Expr)
    (array : ArrayBytes layout native reference profile state tensor shape expressions)
    (instructions : Nat → Instruction)
    (certificates : ∀ i, i<tensor.numel → Rewrite (compile (expressions i)).value (instructions i)) :
    ∀ i, i<tensor.numel → ∀ byte : Fin 4,
      bytes layout (word native profile) state (layout.buffer state.native tensor.region).device
        ((layout.buffer state.native tensor.region).base+4*tensor.index i+byte.val)=
      byteOf (if tensor.isNeg then flipSign (runInstruction native profile (instructions i))
        else runInstruction native profile (instructions i)) byte := by
  intro i hi byte
  have result := array.2.2 i hi byte
  rw [← ports_equal native reference ports] at result
  rw [(certificates i hi).sound native profile laws one] at result
  exact result

theorem forward_graph_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Expr σ) (negation : EncodesNegation (word native profile) obs)
    (rt : Runtime Expr σ) (lib : Library obs hostArithmetic rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World Expr σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (success : Host.forward algebra rt o x y initial=.ok returned final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region) (heldZ : kept returned.2.2.region) :
    ∃ (e : ForwardData algebra rt o x y rows cols initial returned final) (k : ForwardKernelData obs e),
      ArrayBytes layout native reference profile final returned.1 [rows] (finalLoss k hostArithmetic) ∧
      ArrayBytes layout native reference profile final returned.2.1 [rows] (finalLSE k hostArithmetic) ∧
      ArrayBytes layout native reference profile final returned.2.2 [rows] (finalZ k hostArithmetic) := by
  obtain ⟨e⟩ := forward_success algebra rt o x y rows cols initial returned final shape success
  obtain ⟨k⟩ := forward_kernel_data obs hostArithmetic rt lib e xLive yLive
  obtain ⟨loss,lse,z,rest⟩ := forward_values hostArithmetic lib ints k yLive
  have backed := forward_return_backing life physical k loss.dtype lse.dtype z.dtype heldLoss heldLSE heldZ
  exact ⟨e,k,array_bytes layout native reference profile ports obs negation _ _ _ _ loss backed.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ lse backed.2.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ z backed.2.2⟩

/-- Standalone backward uses the caller-supplied LSE exactly. It is not
silently replaced by an independently recomputed normalizer. -/
theorem backward_graph_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Expr σ) (negation : EncodesNegation (word native profile) obs)
    (rt : Runtime Expr σ) (lib : Library obs hostArithmetic rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World Expr σ) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (success : Host.backward algebra rt o g x l y initial=.ok output final) (held : kept output.region) :
    ArrayBytes layout native reference profile final output [rows,cols]
      (fun i => gradientScalar algebra o (obs.integer initial y (i/cols))
        (value obs initial x i) (value obs initial l (i/cols)) (value obs initial g (i/cols)) (i%cols)) := by
  obtain ⟨e⟩ := backward_success algebra rt o g x l y rows cols initial output final shape success
  obtain ⟨k⟩ := backward_kernel_data obs hostArithmetic rt lib e shape xLive yLive gLive lLive
  have array := backward_array k shape
  have backed := backward_return_backing life physical k held
  apply array_bytes layout native reference profile ports obs negation _ _ _ _ _ backed
  refine ⟨array.shapeEq,array.dtype,array.live,?_⟩
  intro i hi
  have size : output.numel=rows*cols := by simp [Tensor.numel,array.shapeEq]
  have slot := VeriTile.Bench.Repairs.CrossEntropy1.Host.flattened_slot rows cols i (by simpa only [size] using hi)
  have graph := backward_graph_lane hostArithmetic lib k shape yLive gLive lLive (i/cols) (i%cols) slot.2.1 slot.1
  rw [slot.2.2] at graph
  exact (array.values i hi).trans graph

/-- Complete returned-byte composition for the explicit source/host graph.
The saved LSE, all allocations, output identities and backing are derived
inside the proof. Native profile matching is a separate structural task. -/
theorem complete_returned_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Expr σ) (negation : EncodesNegation (word native profile) obs)
    (rt : Runtime Expr σ) (lib : Library obs hostArithmetic rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    {o : Options} {g x y : Tensor} {rows cols : Nat} {initial saved final : World Expr σ}
    {returned : Return} {output : Tensor}
    {e : ForwardData algebra rt o x y rows cols initial returned saved} (k : ForwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region)
    (b : BackwardData algebra rt o g x returned.2.1 y rows cols saved output final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    ArrayBytes layout native reference profile saved returned.1 [rows] (finalLoss k hostArithmetic) ∧
    ArrayBytes layout native reference profile saved returned.2.1 [rows] (finalLSE k hostArithmetic) ∧
    ArrayBytes layout native reference profile saved returned.2.2 [rows] (finalZ k hostArithmetic) ∧
    ArrayBytes layout native reference profile final output [rows,cols]
      (fun i => gradientScalar algebra o (obs.integer initial y (i/cols))
        (value obs initial x i) (finalLSE k hostArithmetic (i/cols)) (value obs saved g (i/cols)) (i%cols)) := by
  obtain ⟨loss,lse,z,gradient⟩ := complete_graph obs algebra hostArithmetic rt lib ints k shape xLive yLive gLive b
  have forwardBacking := forward_return_backing life physical k loss.dtype lse.dtype z.dtype heldLoss heldLSE heldZ
  have forwardFrame := (forward_values hostArithmetic lib ints k yLive).2.2.2.2.2.2.2
  obtain ⟨bk⟩ := backward_kernel_data obs hostArithmetic rt lib b shape
    (forwardFrame.live _ xLive) (forwardFrame.live _ yLive) gLive lse.live
  have backwardBacking := backward_return_backing life physical bk heldOutput
  exact ⟨array_bytes layout native reference profile ports obs negation _ _ _ _ loss forwardBacking.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ lse forwardBacking.2.1,
    array_bytes layout native reference profile ports obs negation _ _ _ _ z forwardBacking.2.2,
    array_bytes layout native reference profile ports obs negation _ _ _ _ gradient backwardBacking⟩

#axiomsClean array_bytes
#axiomsClean certified_array_bytes
#axiomsClean forward_graph_bytes
#axiomsClean backward_graph_bytes
#axiomsClean complete_returned_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical
