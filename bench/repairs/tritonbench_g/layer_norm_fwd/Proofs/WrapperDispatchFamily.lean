import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperNativeSuffix
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeGeometry

/-! Configuration is partial and uses the actual predecessor state. Raw
wrapper arguments may have block=0 or inadmissible grids, so they cannot all
be assigned an accepted Config. Successful configurations contain only local
machine/profile and storage facts; execution computes the outputs itself. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

structure NativeLaunch (α : Type) (caller : Address → Prop) (block : Nat) where
  config : Config α block
  benchmarks : List (DispatchEvent block)
  selected : Schedule block
  fresh : CallerFresh config caller
  separated : Domain.ReadWriteSeparated config.flags config.regions
  geometry : NativeGeometry.Geometry config.flags config.regions config.strides config.M config.N

structure NativeFamily (α Host : Type) (caller : Address → Prop) where
  configure : (args : LaunchArguments) → HostState α Host →
    Except Failure (NativeLaunch α caller args.prepared.block)

def NativeLaunch.action (launch : NativeLaunch α caller block) :
    Action (world (α:=α) (Host:=Host) caller) Unit :=
  dispatchAction launch.config caller launch.fresh launch.benchmarks launch.selected

def NativeFamily.kernel (family : NativeFamily α Host caller) (args : LaunchArguments) :
    Action (world (α:=α) (Host:=Host) caller) Unit where
  run s := match family.configure args s with
    | .error e => ⟨s,.error e,[]⟩
    | .ok launch => launch.action.run s
  frame s i hi := by
    cases he : family.configure args s with
    | error e => simp [he]
    | ok launch => exact launch.action.frame s i hi

theorem family_kernel_failure (family : NativeFamily α Host caller) (args : LaunchArguments)
    (s : HostState α Host) (error : Failure) (h : family.configure args s=.error error) :
    (family.kernel args).run s=⟨s,.error error,[]⟩ := by simp [NativeFamily.kernel,h]

theorem family_kernel_success (family : NativeFamily α Host caller) (args : LaunchArguments)
    (s : HostState α Host) (h : ((family.kernel args).run s).result=.ok ()) :
    ∃ launch, family.configure args s=.ok launch ∧
      (launch.action.run s).result=.ok () ∧ (family.kernel args).run s=launch.action.run s := by
  cases he : family.configure args s with
  | error error => simp [NativeFamily.kernel,he] at h
  | ok launch => exact ⟨launch,rfl,by simpa [NativeFamily.kernel,he] using h,
      by simp [NativeFamily.kernel,he]⟩

/-- In particular the interface is consistent on zero-block raw arguments:
they retain a failure outcome instead of demanding an impossible Config. -/
theorem family_zero_block (family : NativeFamily α Host caller) (args : LaunchArguments)
    (s : HostState α Host) (h : args.prepared.block=0) :
    ∃ failure, family.configure args s=.error failure := by
  cases he : family.configure args s with
  | error failure => exact ⟨failure,rfl⟩
  | ok launch =>
    have hp := launch.config.accepted.blockPositive
    omega

#axiomsClean family_kernel_failure
#axiomsClean family_kernel_success
#axiomsClean family_zero_block
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
