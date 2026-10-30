import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Contract
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Invocation

namespace VeriTile.Bench.Repairs.CrossEntropy2.Checked
open VeriTile.Triton Source Host Floating
open Floating.Profiled (DeviceFamily BackendABI BackwardABI BackwardOrder)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

def profiledForward {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ)
    (mx my : TensorMetadata) (ms : ScalarMetadata) (splitBool : Bool)
    (o : Options) (x y : Tensor) (rows cols : Nat) (target : Nat → Int) (s : World Compiled σ) :
    Step (World Compiled σ) (DeviceFamily × Profile × Return) := by
  classical
  exact if ForwardContract mx my ms splitBool o x y rows cols target then Invocation.forward layout abi rt o x y s
    else .error "checked forward contract" s

def profiledBackward {σ : Type} (layout : Layout σ) (abi : BackwardABI) (device : DeviceFamily)
    (rt : Runtime Compiled σ) (mx my mg ml : TensorMetadata) (ms : ScalarMetadata)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (target : Nat → Int) (s : World Compiled σ) :
    Step (World Compiled σ) (BackwardOrder × Tensor) := by
  classical
  exact if BackwardContract mx my mg ml ms o g x l y rows cols target then Invocation.backward layout abi device rt o g x l y s
    else .error "checked backward contract" s

theorem profiled_forward_success {σ : Type} {layout : Layout σ} {abi : BackendABI} {rt : Runtime Compiled σ}
    {mx my : TensorMetadata} {ms : ScalarMetadata} {splitBool : Bool}
    {o : Options} {x y : Tensor} {rows cols : Nat} {target : Nat → Int}
    {initial final : World Compiled σ} {returned : DeviceFamily × Profile × Return} :
    profiledForward layout abi rt mx my ms splitBool o x y rows cols target initial=.ok returned final ↔
      ForwardContract mx my ms splitBool o x y rows cols target ∧ Invocation.forward layout abi rt o x y initial=.ok returned final := by
  classical
  unfold profiledForward
  split_ifs with h <;> simp [h]

theorem profiled_backward_success {σ : Type} {layout : Layout σ} {abi : BackwardABI} {device : DeviceFamily}
    {rt : Runtime Compiled σ} {mx my mg ml : TensorMetadata} {ms : ScalarMetadata}
    {o : Options} {g x l y : Tensor} {rows cols : Nat} {target : Nat → Int}
    {initial final : World Compiled σ} {returned : BackwardOrder × Tensor} :
    profiledBackward layout abi device rt mx my mg ml ms o g x l y rows cols target initial=.ok returned final ↔
      BackwardContract mx my mg ml ms o g x l y rows cols target ∧ Invocation.backward layout abi device rt o g x l y initial=.ok returned final := by
  classical
  unfold profiledBackward
  split_ifs with h <;> simp [h]

#axiomsClean profiled_forward_success
#axiomsClean profiled_backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Checked
