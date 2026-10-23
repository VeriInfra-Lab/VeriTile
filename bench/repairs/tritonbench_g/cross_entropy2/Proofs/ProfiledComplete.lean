import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LaunchContracts
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AllFrames

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical.Profiled
open VeriTile.Triton Source Host Floating
open Floating.Scheduled (decode)
open Floating.Profiled (DeviceFamily BackendABI DispatchLibrary DeviceSelected TorchSelected
  BackwardABI BackwardOrder BackwardLibrary BackwardSelected)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout EncodesNegation)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Complete forward result with actual source-to-profile dispatch evidence.
All profiles are explicit graph data; only scalar numerical laws are deferred. -/
theorem forward_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackendABI)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (dispatch : DispatchLibrary layout abi device host rt)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (success : Host.forward (Floating.Profiled.algebra device o.logitScale) rt o x y initial=.ok returned final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region) (heldZ : kept returned.2.2.region) :
    let profile := Floating.Profiled.profile device host
    ∃ (e : ForwardData (Floating.Profiled.algebra device o.logitScale) rt o x y rows cols initial returned final)
      (k : ForwardKernelData obs e),
      DeviceSelected layout abi device e ∧ TorchSelected layout abi host e ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows e.count o.width cols) ∧
      Frames.Frame obs layout kept initial final ∧
      NativeArrayBytes layout n profile final returned.1 [rows] (Floating.Profiled.nativeFinalLoss n profile k) ∧
      NativeArrayBytes layout n profile final returned.2.1 [rows] (Floating.Profiled.nativeFinalLSE n profile k) ∧
      NativeArrayBytes layout n profile final returned.2.2 [rows] (Floating.Profiled.nativeFinalZ n profile k) := by
  obtain ⟨e,k,bytes⟩ := forward_native_bytes layout n reference (Floating.Profiled.profile device host) ports laws one
    kept obs negation rt lib ints life physical device o x y rows cols initial final returned shape xLive yLive success
    heldLoss heldLSE heldZ
  have selected := Floating.Profiled.source_profiles_selected dispatch k
  have frame := Frames.forward_frame life ints (Floating.Profiled.algebra device o.logitScale) o x y initial
  rw [success] at frame
  exact ⟨e,k,selected.1,selected.2,Native.forward_supported grid e,frame,bytes⟩

/-- Standalone backward uses the supplied LSE and records the actual local
compiler order selected for that call, including the skipped empty grid. -/
theorem backward_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackwardABI) (order : BackwardOrder)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (dispatch : BackwardLibrary layout abi order rt)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World Compiled σ) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (success : Host.backward (Floating.Profiled.algebra device o.logitScale) rt o g x l y initial=.ok output final)
    (held : kept output.region) :
    let profile := Floating.Profiled.profile device host
    ∃ e : BackwardData (Floating.Profiled.algebra device o.logitScale) rt o g x l y rows cols initial output final,
      BackwardSelected layout abi order e ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows e.count o.width cols) ∧
      Frames.Frame obs layout kept initial final ∧
      NativeArrayBytes layout n profile final output [rows,cols]
        (fun i => nativeGradient n o (obs.integer initial y (i/cols))
          (decode n profile (value obs initial x i)) (decode n profile (value obs initial l (i/cols)))
          (decode n profile (value obs initial g (i/cols))) (i%cols) (order.fma i) (order.output i)) := by
  obtain ⟨e⟩ := backward_success (Floating.Profiled.algebra device o.logitScale) rt o g x l y rows cols initial output final shape success
  have bytes := backward_native_bytes layout n reference (Floating.Profiled.profile device host) ports laws one
    kept obs negation rt lib life physical device o g x l y rows cols initial final output order.fma order.output
    shape xLive yLive gLive lLive success held
  have frame := Frames.backward_frame life ints (Floating.Profiled.algebra device o.logitScale) o g x l y initial
  rw [success] at frame
  exact ⟨e,Floating.Profiled.source_backward_selected dispatch e,Native.backward_supported grid e,frame,bytes⟩

/-- Whole immediate forward/backward call. The source selects every actual
local compiler and Torch profile from its prepared metadata; the full
instruction graph then reaches every returned byte. All caller frames and
the actual saved LSE are retained. No fixture width or smoothing/split flag
is imposed, and no reduction/output value is a numerical premise. -/
theorem complete_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackendABI) (backwardABI : BackwardABI) (order : BackwardOrder)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (dispatch : DispatchLibrary layout abi device host rt) (backwardDispatch : BackwardLibrary layout backwardABI order rt)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World Compiled σ)
    (returned : Return) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region) (gLive : saved.live g.region)
    (forwardSuccess : Host.forward (Floating.Profiled.algebra device o.logitScale) rt o x y initial=.ok returned saved)
    (backwardSuccess : Host.backward (Floating.Profiled.algebra device o.logitScale) rt o g x returned.2.1 y saved=.ok output final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    let profile := Floating.Profiled.profile device host
    ∃ (e : ForwardData (Floating.Profiled.algebra device o.logitScale) rt o x y rows cols initial returned saved)
      (k : ForwardKernelData obs e)
      (b : BackwardData (Floating.Profiled.algebra device o.logitScale) rt o g x returned.2.1 y rows cols saved output final),
      DeviceSelected layout abi device e ∧ TorchSelected layout abi host e ∧ BackwardSelected layout backwardABI order b ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows e.count o.width cols) ∧
      (rows=0 ∨ VeriTile.Bench.Repairs.CrossEntropy1.Native.Supported rows b.count o.width cols) ∧
      Frames.Frame obs layout kept initial saved ∧ Frames.Frame obs layout kept saved final ∧
      NativeArrayBytes layout n profile saved returned.1 [rows] (Floating.Profiled.nativeFinalLoss n profile k) ∧
      NativeArrayBytes layout n profile saved returned.2.1 [rows] (Floating.Profiled.nativeFinalLSE n profile k) ∧
      NativeArrayBytes layout n profile saved returned.2.2 [rows] (Floating.Profiled.nativeFinalZ n profile k) ∧
      NativeArrayBytes layout n profile final output [rows,cols]
        (fun i => nativeGradient n o (obs.integer initial y (i/cols))
          (decode n profile (value obs initial x i)) (Floating.Profiled.nativeFinalLSE n profile k (i/cols))
          (decode n profile (value obs saved g (i/cols))) (i%cols) (order.fma i) (order.output i)) := by
  obtain ⟨e,k,bytes⟩ := complete_native_bytes layout n reference (Floating.Profiled.profile device host) ports laws one
    kept obs negation rt lib ints life physical device o g x y rows cols initial saved final returned output order.fma order.output
    shape xLive yLive gLive forwardSuccess backwardSuccess heldLoss heldLSE heldZ heldOutput
  obtain ⟨b⟩ := backward_success (Floating.Profiled.algebra device o.logitScale) rt o g x returned.2.1 y rows cols saved output final shape backwardSuccess
  have selected := Floating.Profiled.source_profiles_selected dispatch k
  have ff := Frames.forward_frame life ints (Floating.Profiled.algebra device o.logitScale) o x y initial
  have bf := Frames.backward_frame life ints (Floating.Profiled.algebra device o.logitScale) o g x returned.2.1 y saved
  rw [forwardSuccess] at ff
  rw [backwardSuccess] at bf
  exact ⟨e,k,b,selected.1,selected.2,Floating.Profiled.source_backward_selected backwardDispatch b,
    Native.forward_supported grid e,Native.backward_supported grid b,ff,bf,bytes⟩

#axiomsClean forward_profiled_bytes
#axiomsClean backward_profiled_bytes
#axiomsClean complete_profiled_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical.Profiled
