import bench.repairs.tritonbench_g.cross_entropy2.Proofs.CheckedInvocation
import bench.repairs.tritonbench_g.cross_entropy2.Kernels
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.NativeReturns
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.NativeReadback
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScheduledReadback
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.CompilerSites
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LargeProfile
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AlignedProfile
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.TorchButterfly
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AllFrames
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.RoleBindings
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfiledComplete
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfileInstances
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationReadback
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationCoexist

namespace VeriTile.Bench.Repairs.CrossEntropy2.FPEquiv
open VeriTile.Triton Source Host Floating Physical
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout EncodesNegation)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
open Floating.Scheduled (decode)
open Floating.Profiled (DeviceFamily BackendABI DeviceSelected TorchSelected
  BackwardABI BackwardOrder BackwardSelected)
noncomputable section

/-- Production precision: every floating input, scalar port, accumulator,
intermediate and returned tensor is FP32. Labels are signed int32; class
indices and address intermediates are int64. Default FMA fusion is retained.
All FP32 input bit patterns, including NaNs, infinities and signed zero, remain
in scope. The scalar conformance laws are explicit conditional hypotheses. -/
abbrev Domain := Host.valid
abbrev NumericalObligations := Floating.Obligations
abbrev CompilerScalarObligations := Floating.Peepholes

/-- Whole immediate forward/backward call with invocation-local backend
selection. Each actual setup/library call selects its own graph from actual
metadata. Forgetting graph annotations recovers the same original source
traces; backward retains the forward-selected decoder and actual saved LSE.
No globally fixed profile/metadata-equality contract is a premise. -/
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
          (decode n profile (value obs saved g (i/cols))) (i%cols) (order.fma i) (order.output i)) :=
  Physical.Invocation.complete_profiled_bytes layout n reference device host abi backwardABI order ports laws one
    kept obs negation rt lib ints grid life physical o g x y rows cols initial saved final returned output
    shape xLive yLive gLive forwardSuccess backwardSuccess heldLoss heldLSE heldZ heldOutput
theorem checked_complete_profiled_bytes {σ : Type} (layout : Layout σ) (n reference : Ports)
    (device : DeviceFamily) (host : Profile) (abi : BackendABI) (backwardABI : BackwardABI) (order : BackwardOrder)
    (ports : Obligations n reference) (laws : Peepholes n) (one : ConstantOne n)
    (kept : RegionName → Prop) (obs : Observations Compiled σ)
    (negation : EncodesNegation (decode n (Floating.Profiled.profile device host)) obs)
    (rt : Runtime Compiled σ) (lib : Library obs Floating.Scheduled.hostArithmetic rt) (ints : IntegerViews obs)
    (grid : Native.GridLibrary rt)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (mx my mg ml : Checked.TensorMetadata) (ms : Checked.ScalarMetadata) (splitBool : Bool)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World Compiled σ)
    (returned : Return) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region) (gLive : saved.live g.region)
    (forwardSuccess : Checked.profiledForward layout abi rt mx my ms splitBool o x y rows cols (obs.integer initial y) initial=.ok (device,host,returned) saved)
    (backwardSuccess : Checked.profiledBackward layout backwardABI device rt mx my mg ml ms o g x returned.2.1 y rows cols (obs.integer saved y) saved=.ok (order,output) final)
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
          (decode n profile (value obs saved g (i/cols))) (i%cols) (order.fma i) (order.output i)) :=
  Physical.Invocation.complete_profiled_bytes layout n reference device host abi backwardABI order ports laws one
    kept obs negation rt lib ints grid life physical o g x y rows cols initial saved final returned output
    shape xLive yLive gLive (Checked.profiled_forward_success.mp forwardSuccess).2
    (Checked.profiled_backward_success.mp backwardSuccess).2 heldLoss heldLSE heldZ heldOutput

#axiomsClean checked_complete_profiled_bytes

/-- The backend APIs return structural graphs/order, never numerical outputs.
They are used dynamically inside Invocation.forward/backward. -/
abbrev BackendGraphAPI := Floating.Profiled.BackendABI
abbrev BackwardGraphAPI := Floating.Profiled.BackwardABI
abbrev LaunchContracts := @Native.GridLibrary

export Physical.Invocation (forward_profiled_bytes backward_profiled_bytes)
export Invocation (forward_success backward_success same_width_different_graphs launch_uses_actual_key two_calls_coexist)
export Floating.Profiled (pure_smoothing_exact alignedPlan offset4Plan)
export Native (forward_supported backward_supported)

/-- Public complete source/host graph byte theorem. It extracts both actual
host traces and the kernel preparation; no final-output or LSE premise is
accepted. Explicit seeded trees and scalar ports define the FP reference.
This retained pure-tree helper predates the general profiled entry above.
`complete_profiled_bytes` includes actual local compiler/Torch selection;
`aligned_bound_bytes` retains the concrete recorded aligned JIT-key witness. -/
theorem complete_graph_bytes {σ : Type} (layout : Layout σ) (native reference : Ports) (profile : Profile)
    (ports : NumericalObligations native reference) (kept : RegionName → Prop)
    (obs : Observations Expr σ) (negation : EncodesNegation (word native profile) obs)
    (rt : Runtime Expr σ) (lib : Library obs hostArithmetic rt) (ints : IntegerViews obs)
    (life : Frames.LocalLibrary obs layout kept rt) (physical : BackingLibrary layout rt)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World Expr σ)
    (returned : Return) (output : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region)
    (forwardSuccess : Host.forward algebra rt o x y initial=.ok returned saved)
    (backwardSuccess : Host.backward algebra rt o g x returned.2.1 y saved=.ok output final)
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (heldZ : kept returned.2.2.region) (heldOutput : kept output.region) :
    ∃ (e : ForwardData algebra rt o x y rows cols initial returned saved) (k : ForwardKernelData obs e),
      ArrayBytes layout native reference profile saved returned.1 [rows] (finalLoss k hostArithmetic) ∧
      ArrayBytes layout native reference profile saved returned.2.1 [rows] (finalLSE k hostArithmetic) ∧
      ArrayBytes layout native reference profile saved returned.2.2 [rows] (finalZ k hostArithmetic) ∧
      ArrayBytes layout native reference profile final output [rows,cols]
        (fun i => gradientScalar algebra o (obs.integer initial y (i/cols))
          (value obs initial x i) (finalLSE k hostArithmetic (i/cols)) (value obs saved g (i/cols)) (i%cols)) := by
  obtain ⟨e⟩ := Host.forward_success algebra rt o x y rows cols initial returned saved shape forwardSuccess
  obtain ⟨k⟩ := forward_kernel_data obs hostArithmetic rt lib e xLive yLive
  obtain ⟨b⟩ := Host.backward_success algebra rt o g x returned.2.1 y rows cols saved output final shape backwardSuccess
  exact ⟨e,k,complete_returned_bytes layout native reference profile ports kept obs negation rt lib ints life physical
    k shape xLive yLive gLive b heldLoss heldLSE heldZ heldOutput⟩

/- Separate scalar compiler identities. These include the actual negated
factor exchange in the unsmoothed target, FMA factor swaps, doubled scale,
saved z multiplication and stable-LSE add ordering. They are conditional
local numerical laws, never complete reductions or kernel equivalence ports. -/
export Floating (scaled_target_site scaled_logit_site saved_z_site stable_lse_site host_scale_site
  backward_correction_site squared_lse_site smoothed_target_site subtract_one_site exp_site)
/- Complete selected scalar/host composition: guarded loads and reductions, every
loss/z branch, and the standalone or composed gradient reach actual return
bytes. These retained pure-tree helpers remain useful for their scoped
profiles. The whole-domain profiled entry above records actual source calls,
metadata and graph-valued backend selections, including reduction-level FMA. -/
export Floating (forward_reductions_native loss_z_native gradient_native role_native final_native)
export Floating (aligned_smoothing_covers aligned_smoothing_exact aligned_shared_mask)
export Floating.Scheduled (source_smoothing_scalar source_lse_scalar aligned_role_scalar aligned_metadata_mask)
export Physical.Scheduled (aligned_bound_bytes)
export Floating (thread_four_exact thread_indices_cover partition_exact down_exact torch_reduced_axis_dominates)
export Invocation (forward_frame backward_frame)
export Frames (Frame.all_bytes)
export Physical (forward_roles forward_target_role backward_roles forward_graph_bytes backward_graph_bytes certified_array_bytes)
export Physical (NativeArrayBytes forward_native_bytes backward_native_bytes complete_native_bytes)

#axiomsClean complete_graph_bytes
#axiomsClean complete_profiled_bytes
/- The registry-only `#print_fp_assumptions` command does not accept byte
readback theorems. Print the complete explicit scalar interfaces instead. -/
#print Floating.Obligations
#print Floating.Peepholes
#print Floating.ConstantOne
#print Floating.ExpImplementation
/- Structural backend interfaces are printed separately from numerical laws. -/
#print Floating.Profiled.BackendABI
#print Floating.Profiled.BackwardABI
#print Native.GridLibrary
#specNonCircular NumericalObligations avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular CompilerScalarObligations avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular BackendGraphAPI avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular BackwardGraphAPI avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular LaunchContracts avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.FPEquiv
