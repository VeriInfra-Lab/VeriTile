import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationBackward
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationFrames

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical.Invocation
open VeriTile.Triton Source Host Floating
open Floating.Scheduled (decode)
open Floating.Profiled (DeviceFamily BackendABI DeviceSelected TorchSelected BackwardABI BackwardOrder BackwardSelected)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout EncodesNegation)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

private theorem native_array_bytes {σ : Type} (layout : Layout σ) (n reference : Ports) (profile : Profile)
    (ports : Obligations n reference) (state : World Compiled σ) (tensor : Tensor) (shape : List Nat)
    (expressions : Nat → Compiled) (values : Nat → Word)
    (array : Physical.Profiled.ArrayBytes layout n reference profile state tensor shape expressions)
    (derived : ∀ i, decode n profile (expressions i)=values i) :
    Physical.Profiled.NativeArrayBytes layout n profile state tensor shape values := by
  refine ⟨array.1,array.2.1,?_⟩
  intro i hi byte
  have result := array.2.2 i hi byte
  rw [← ports_equal n reference ports,derived i] at result
  exact result


theorem forward_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackendABI)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (success : Invocation.forward layout abi rt o x y initial=.ok (device,host,returned) final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region) (heldZ : kept returned.2.2.region) :
    let profile := Floating.Profiled.profile device host
    ∃ (e : ForwardData (Floating.Profiled.algebra device o.logitScale) rt o x y rows cols initial returned final)
      (k : ForwardKernelData obs e),
      DeviceSelected layout abi device e ∧ TorchSelected layout abi host e ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows e.count o.width cols) ∧
      Frames.Frame obs layout kept initial final ∧
      Physical.Profiled.NativeArrayBytes layout n profile final returned.1 [rows] (Floating.Profiled.nativeFinalLoss n profile k) ∧
      Physical.Profiled.NativeArrayBytes layout n profile final returned.2.1 [rows] (Floating.Profiled.nativeFinalLSE n profile k) ∧
      Physical.Profiled.NativeArrayBytes layout n profile final returned.2.2 [rows] (Floating.Profiled.nativeFinalZ n profile k) := by
  obtain ⟨e,source,ds,ts⟩ := Invocation.forward_success layout abi rt o x y rows cols initial final device host returned shape success
  obtain ⟨k⟩ := forward_kernel_data obs Floating.Scheduled.hostArithmetic rt lib e xLive yLive
  obtain ⟨loss,lse,z,rest⟩ := forward_values Floating.Scheduled.hostArithmetic lib ints k yLive
  have backed := forward_return_backing life physical k loss.dtype lse.dtype z.dtype heldLoss heldLSE heldZ
  have frame := Invocation.forward_frame life ints abi o x y initial
  rw [success] at frame
  refine ⟨e,k,ds,ts,Native.forward_supported grid e,frame,?_,?_,?_⟩
  · apply native_array_bytes layout n reference _ ports _ _ _ _ _
      (Physical.Profiled.array_bytes layout n reference _ ports obs negation _ _ _ _ loss backed.1)
    intro i
    exact (Floating.Profiled.final_native n _ laws one k i).1
  · apply native_array_bytes layout n reference _ ports _ _ _ _ _
      (Physical.Profiled.array_bytes layout n reference _ ports obs negation _ _ _ _ lse backed.2.1)
    intro i
    exact (Floating.Profiled.final_native n _ laws one k i).2.1
  · apply native_array_bytes layout n reference _ ports _ _ _ _ _
      (Physical.Profiled.array_bytes layout n reference _ ports obs negation _ _ _ _ z backed.2.2)
    intro i
    exact (Floating.Profiled.final_native n _ laws one k i).2.2

theorem backward_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackwardABI) (order : BackwardOrder)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (success : Invocation.backward layout abi device rt o g x l y initial=.ok (order,output) final)
    (held : kept output.region) :
    let profile := Floating.Profiled.profile device host
    ∃ e : BackwardData (Floating.Profiled.algebra device o.logitScale) rt o g x l y rows cols initial output final,
      BackwardSelected layout abi order e ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows e.count o.width cols) ∧
      Frames.Frame obs layout kept initial final ∧
      Physical.Profiled.NativeArrayBytes layout n profile final output [rows,cols]
        (fun i => nativeGradient n o (obs.integer initial y (i/cols))
          (decode n profile (value obs initial x i)) (decode n profile (value obs initial l (i/cols)))
          (decode n profile (value obs initial g (i/cols))) (i%cols) (order.fma i) (order.output i)) := by
  obtain ⟨e,source,selected⟩ := Invocation.backward_success layout abi device rt o g x l y rows cols initial final order output shape success
  have bytes := Physical.Profiled.backward_native_bytes layout n reference (Floating.Profiled.profile device host) ports laws one
    kept obs negation rt lib life physical device o g x l y rows cols initial final output order.fma order.output
    shape xLive yLive gLive lLive source held
  have frame := Invocation.backward_frame life ints abi device o g x l y initial
  rw [success] at frame
  exact ⟨e,selected,Native.backward_supported grid e,frame,bytes⟩

theorem complete_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackendABI) (backwardABI : BackwardABI) (order : BackwardOrder)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World Compiled σ)
    (returned : Return) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region) (gLive : saved.live g.region)
    (forwardSuccess : Invocation.forward layout abi rt o x y initial=.ok (device,host,returned) saved)
    (backwardSuccess : Invocation.backward layout backwardABI device rt o g x returned.2.1 y saved=.ok (order,output) final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    let profile := Floating.Profiled.profile device host
    ∃ (e : ForwardData (Floating.Profiled.algebra device o.logitScale) rt o x y rows cols initial returned saved)
      (k : ForwardKernelData obs e)
      (b : BackwardData (Floating.Profiled.algebra device o.logitScale) rt o g x returned.2.1 y rows cols saved output final),
      Host.forward (Floating.Profiled.algebra device o.logitScale) rt o x y initial=.ok returned saved ∧
      Host.backward (Floating.Profiled.algebra device o.logitScale) rt o g x returned.2.1 y saved=.ok output final ∧
      DeviceSelected layout abi device e ∧ TorchSelected layout abi host e ∧ BackwardSelected layout backwardABI order b ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows e.count o.width cols) ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows b.count o.width cols) ∧
      Frames.Frame obs layout kept initial saved ∧ Frames.Frame obs layout kept saved final ∧
      Physical.Profiled.NativeArrayBytes layout n profile saved returned.1 [rows] (Floating.Profiled.nativeFinalLoss n profile k) ∧
      Physical.Profiled.NativeArrayBytes layout n profile saved returned.2.1 [rows] (Floating.Profiled.nativeFinalLSE n profile k) ∧
      Physical.Profiled.NativeArrayBytes layout n profile saved returned.2.2 [rows] (Floating.Profiled.nativeFinalZ n profile k) ∧
      Physical.Profiled.NativeArrayBytes layout n profile final output [rows,cols]
        (fun i => nativeGradient n o (obs.integer initial y (i/cols))
          (decode n profile (value obs initial x i)) (Floating.Profiled.nativeFinalLSE n profile k (i/cols))
          (decode n profile (value obs saved g (i/cols))) (i%cols) (order.fma i) (order.output i)) := by
  obtain ⟨e,forwardSource,ds,ts⟩ := Invocation.forward_success layout abi rt o x y rows cols initial saved device host returned shape forwardSuccess
  obtain ⟨b,backwardSource,bs⟩ := Invocation.backward_success layout backwardABI device rt o g x returned.2.1 y rows cols saved final order output shape backwardSuccess
  obtain ⟨k⟩ := forward_kernel_data obs Floating.Scheduled.hostArithmetic rt lib e xLive yLive
  obtain ⟨loss,lse,z,gradient⟩ := Physical.Profiled.complete_returned_bytes layout n reference (Floating.Profiled.profile device host) ports
    kept obs negation rt lib ints life physical device k shape xLive yLive gLive b heldLoss heldLSE heldZ heldOutput
  have ff := Invocation.forward_frame life ints abi o x y initial
  have bf := Invocation.backward_frame life ints backwardABI device o g x returned.2.1 y saved
  rw [forwardSuccess] at ff
  rw [backwardSuccess] at bf
  refine ⟨e,k,b,forwardSource,backwardSource,ds,ts,bs,Native.forward_supported grid e,Native.backward_supported grid b,ff,bf,?_,?_,?_,?_⟩
  · exact native_array_bytes layout n reference _ ports _ _ _ _ _ loss
      (fun i => (Floating.Profiled.final_native n _ laws one k i).1)
  · exact native_array_bytes layout n reference _ ports _ _ _ _ _ lse
      (fun i => (Floating.Profiled.final_native n _ laws one k i).2.1)
  · exact native_array_bytes layout n reference _ ports _ _ _ _ _ z
      (fun i => (Floating.Profiled.final_native n _ laws one k i).2.2)
  · apply native_array_bytes layout n reference _ ports _ _ _ _ _ gradient
    intro i
    rw [Floating.Profiled.gradient_scalar device n _ laws one o _ _ _ _ _ (order.fma i) (order.output i)]
    rw [(Floating.Profiled.final_native n _ laws one k (i/cols)).2.1]

#axiomsClean forward_profiled_bytes
#axiomsClean backward_profiled_bytes
#axiomsClean complete_profiled_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical.Invocation
