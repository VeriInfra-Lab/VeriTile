import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledFinal
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledReturns

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical.Scheduled
open VeriTile.Triton Source Host Floating
open Floating.Scheduled (decode)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout EncodesNegation bytes byteOf flipSign)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Byte readback of the aligned scalar and reduction graph, including the
three-FMA prefix and its shared select. Matching keys are stated separately
in Specialization; general compiler/Torch profiles are not asserted here. -/
def NativeArrayBytes {σ : Type} (layout : Layout σ) (n : Ports) (profile : Profile)
    (state : World Compiled σ) (tensor : Tensor) (shape : List Nat) (values : Nat → Word) : Prop :=
  tensor.shape=shape ∧ tensor.dtype="torch.float32" ∧
  ∀ i, i<tensor.numel → ∀ byte : Fin 4,
    bytes layout (decode n profile) state (layout.buffer state.native tensor.region).device
      ((layout.buffer state.native tensor.region).base+4*tensor.index i+byte.val)=
    byteOf (if tensor.isNeg then flipSign (values i) else values i) byte

private theorem native_array_bytes {σ : Type} (layout : Layout σ) (n reference : Ports) (profile : Profile)
    (ports : Obligations n reference) (state : World Compiled σ) (tensor : Tensor) (shape : List Nat)
    (expressions : Nat → Compiled) (values : Nat → Word)
    (array : ArrayBytes layout n reference profile state tensor shape expressions)
    (derived : ∀ i, decode n profile (expressions i)=values i) :
    NativeArrayBytes layout n profile state tensor shape values := by
  refine ⟨array.1,array.2.1,?_⟩
  intro i hi byte
  have result := array.2.2 i hi byte
  rw [← ports_equal n reference ports,derived i] at result
  exact result

theorem forward_native_bytes {σ : Type} (layout : Layout σ) (n reference : Ports) (profile : Profile)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ) (negation : EncodesNegation (decode n profile) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (width : o.width=65536) (smoothing : o.hasSmoothing=Bool.true) (x y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (success : Host.forward (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o x y initial=.ok returned final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region) (heldZ : kept returned.2.2.region) :
    ∃ (e : ForwardData (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o x y rows cols initial returned final) (k : ForwardKernelData obs e),
      NativeArrayBytes layout n profile final returned.1 [rows] (Floating.Scheduled.nativeFinalLoss n profile k) ∧
      NativeArrayBytes layout n profile final returned.2.1 [rows] (Floating.Scheduled.nativeFinalLSE n profile k) ∧
      NativeArrayBytes layout n profile final returned.2.2 [rows] (Floating.Scheduled.nativeFinalZ n profile k) := by
  obtain ⟨e,k,loss,lse,z⟩ := forward_graph_bytes layout n reference profile ports kept obs negation
    rt lib ints life physical .aligned65536 o x y rows cols initial final returned shape xLive yLive success heldLoss heldLSE heldZ
  exact ⟨e,k,native_array_bytes layout n reference profile ports _ _ _ _ _ loss
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).1),
    native_array_bytes layout n reference profile ports _ _ _ _ _ lse
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).2.1),
    native_array_bytes layout n reference profile ports _ _ _ _ _ z
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).2.2)⟩

/-- Standalone backward readback retains the caller's supplied LSE, all
logit/upstream loads, and each local compiler operand-order choice. -/
theorem backward_native_bytes {σ : Type} (layout : Layout σ) (n reference : Ports) (profile : Profile)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ) (negation : EncodesNegation (decode n profile) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (output : Tensor)
    (commutedFMA commutedOutput : Nat → Bool)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (success : Host.backward (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o g x l y initial=.ok output final) (held : kept output.region) :
    NativeArrayBytes layout n profile final output [rows,cols]
      (fun i => nativeGradient n o (obs.integer initial y (i/cols))
        (decode n profile (value obs initial x i)) (decode n profile (value obs initial l (i/cols)))
        (decode n profile (value obs initial g (i/cols))) (i%cols) (commutedFMA i) (commutedOutput i)) := by
  have array := backward_graph_bytes layout n reference profile ports kept obs negation rt lib life physical
    .aligned65536 o g x l y rows cols initial final output shape xLive yLive gLive lLive success held
  apply native_array_bytes layout n reference profile ports _ _ _ _ _ array
  intro i
  exact Floating.Scheduled.gradient_scalar n profile laws one .aligned65536 o _ _ _ _ _ _ _

/-- The complete selected scalar graph reaches every returned byte. The
backward LSE is the actual forward result, after the actual host reductions.
All expression equalities used here are derived from individual scalar laws;
there is no whole-reduction or output equality assumption. -/
theorem complete_native_bytes {σ : Type} (layout : Layout σ) (n reference : Ports) (profile : Profile)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ) (negation : EncodesNegation (decode n profile) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (width : o.width=65536) (smoothing : o.hasSmoothing=Bool.true) (g x y : Tensor) (rows cols : Nat) (initial saved final : World Compiled σ)
    (returned : Return) (output : Tensor) (commutedFMA commutedOutput : Nat → Bool)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region)
    (forwardSuccess : Host.forward (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o x y initial=.ok returned saved)
    (backwardSuccess : Host.backward (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o g x returned.2.1 y saved=.ok output final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    ∃ (e : ForwardData (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o x y rows cols initial returned saved) (k : ForwardKernelData obs e),
      NativeArrayBytes layout n profile saved returned.1 [rows] (Floating.Scheduled.nativeFinalLoss n profile k) ∧
      NativeArrayBytes layout n profile saved returned.2.1 [rows] (Floating.Scheduled.nativeFinalLSE n profile k) ∧
      NativeArrayBytes layout n profile saved returned.2.2 [rows] (Floating.Scheduled.nativeFinalZ n profile k) ∧
      NativeArrayBytes layout n profile final output [rows,cols]
        (fun i => nativeGradient n o (obs.integer initial y (i/cols))
          (decode n profile (value obs initial x i)) (Floating.Scheduled.nativeFinalLSE n profile k (i/cols))
          (decode n profile (value obs saved g (i/cols))) (i%cols) (commutedFMA i) (commutedOutput i)) := by
  obtain ⟨e⟩ := forward_success (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o x y rows cols initial returned saved shape forwardSuccess
  obtain ⟨k⟩ := forward_kernel_data obs Floating.Scheduled.hostArithmetic rt lib e xLive yLive
  obtain ⟨b⟩ := backward_success (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o g x returned.2.1 y rows cols saved output final shape backwardSuccess
  obtain ⟨loss,lse,z,gradient⟩ := complete_returned_bytes layout n reference profile ports kept obs negation rt lib ints
    life physical .aligned65536 k shape xLive yLive gLive b heldLoss heldLSE heldZ heldOutput
  refine ⟨e,k,native_array_bytes layout n reference profile ports _ _ _ _ _ loss
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).1),
    native_array_bytes layout n reference profile ports _ _ _ _ _ lse
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).2.1),
    native_array_bytes layout n reference profile ports _ _ _ _ _ z
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).2.2),?_⟩
  apply native_array_bytes layout n reference profile ports _ _ _ _ _ gradient
  intro i
  rw [Floating.Scheduled.gradient_scalar n profile laws one .aligned65536 o _ _ _ _ _ (commutedFMA i) (commutedOutput i)]
  rw [(Floating.Scheduled.final_native n profile laws one k width smoothing (i/cols)).2.1]

/-- Review entry tying actual prepared pointer/shape metadata to the pinned
compiler rule, concrete device trees, fused smoothing graph, local scalar
interpretation, host aggregation and all returned bytes. Host tree selection
is explicit profile data; this does not assert all-shape Torch conformance. -/
theorem aligned_bound_bytes {σ : Type} (layout : Layout σ) (n reference : Ports) (host : Profile)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (actual : Floating.Scheduled.JITKey → Floating.Scheduled.DeviceSchedule)
    (compiler : Floating.Scheduled.CompilerSelection actual)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Scheduled.deviceTrees .aligned65536 host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    {o : Options} {g x y : Tensor} {rows cols : Nat} {initial saved final : World Compiled σ}
    {returned : Return} {output : Tensor}
    {e : ForwardData (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o x y rows cols initial returned saved}
    (k : ForwardKernelData obs e)
    (key : Floating.Scheduled.keyFor layout k.ready e.parameters=Floating.Scheduled.alignedKey)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region)
    (b : BackwardData (Floating.Scheduled.algebra .aligned65536 o.logitScale) rt o g x returned.2.1 y rows cols saved output final)
    (commutedFMA commutedOutput : Nat → Bool)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    let profile := Floating.Scheduled.deviceTrees .aligned65536 host
    actual (Floating.Scheduled.keyFor layout k.ready e.parameters)=.aligned65536 ∧
    NativeArrayBytes layout n profile saved returned.1 [rows] (Floating.Scheduled.nativeFinalLoss n profile k) ∧
    NativeArrayBytes layout n profile saved returned.2.1 [rows] (Floating.Scheduled.nativeFinalLSE n profile k) ∧
    NativeArrayBytes layout n profile saved returned.2.2 [rows] (Floating.Scheduled.nativeFinalZ n profile k) ∧
    NativeArrayBytes layout n profile final output [rows,cols]
      (fun i => nativeGradient n o (obs.integer initial y (i/cols))
        (decode n profile (value obs initial x i)) (Floating.Scheduled.nativeFinalLSE n profile k (i/cols))
        (decode n profile (value obs saved g (i/cols))) (i%cols) (commutedFMA i) (commutedOutput i)) := by
  let profile := Floating.Scheduled.deviceTrees .aligned65536 host
  have fields := Floating.Scheduled.aligned_key_fields layout k.ready e.parameters key
  have width : o.width=65536 := fields.1
  have smoothing : o.hasSmoothing=Bool.true := fields.2.1
  have chosen : actual (Floating.Scheduled.keyFor layout k.ready e.parameters)=.aligned65536 := by
    rw [key]
    exact compiler.aligned
  obtain ⟨loss,lse,z,gradient⟩ := complete_returned_bytes layout n reference profile ports kept obs negation rt lib ints
    life physical .aligned65536 k shape xLive yLive gLive b heldLoss heldLSE heldZ heldOutput
  refine ⟨chosen,native_array_bytes layout n reference profile ports _ _ _ _ _ loss
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).1),
    native_array_bytes layout n reference profile ports _ _ _ _ _ lse
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).2.1),
    native_array_bytes layout n reference profile ports _ _ _ _ _ z
      (fun i => (Floating.Scheduled.final_native n profile laws one k width smoothing i).2.2),?_⟩
  apply native_array_bytes layout n reference profile ports _ _ _ _ _ gradient
  intro i
  rw [Floating.Scheduled.gradient_scalar n profile laws one .aligned65536 o _ _ _ _ _ (commutedFMA i) (commutedOutput i)]
  rw [(Floating.Scheduled.final_native n profile laws one k width smoothing (i/cols)).2.1]

#axiomsClean aligned_bound_bytes
#axiomsClean forward_native_bytes
#axiomsClean backward_native_bytes
#axiomsClean complete_native_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical.Scheduled
