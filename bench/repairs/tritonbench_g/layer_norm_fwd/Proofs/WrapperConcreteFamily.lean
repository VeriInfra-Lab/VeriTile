import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperCallerReady

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

/-- Compiler/runtime schedule data, including benchmark setup failures.
This supplies no memory, output, index or successful-execution equation. -/
structure DispatchSchedules (block : Nat) where
  benchmarks : List (DispatchEvent block)
  selected : Schedule block

structure ScheduleSelection (Host : Type) where
  select : (args : LaunchArguments) → Host → Except Failure (DispatchSchedules args.prepared.block)

/-- A concrete partial family. Its raw-argument guard is proved true on every
actual admitted prefix in the selected storage profile. Its rejection branch
is not used to classify source-admitted calls in other storage profiles. -/
def preparedFamily (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α) (selection : ScheduleSelection Host) :
    NativeFamily α Host caller where
  configure args current := do
    let schedules ← selection.select args current.host
    if h : PreparedReady store args current caller then
      pure (preparedLaunch store ops encoding args current h schedules.benchmarks schedules.selected)
    else .error (.api .admission 0)

theorem preparedFamily_configured
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α) (selection : ScheduleSelection Host)
    (args : LaunchArguments) (current : HostState α Host)
    (h : PreparedReady store args current caller) (schedules : DispatchSchedules args.prepared.block)
    (hselect : selection.select args current.host=.ok schedules) :
    (preparedFamily store ops encoding selection).configure args current=
      .ok (preparedLaunch store ops encoding args current h schedules.benchmarks schedules.selected) := by
  simp [preparedFamily,hselect,h]

/-- Configuration success follows from the host prefix and successful compiler
schedule selection, before any benchmark or selected kernel is executed. -/
theorem prefix_family_configured
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α) (selection : ScheduleSelection Host)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial)
    (p : Prepared) (args : LaunchArguments) (route : LaunchPrefix tensor device q initial p args)
    (hprofile : SelectedStorage q p) (schedules : DispatchSchedules args.prepared.block)
    (hselect : selection.select args (launchState tensor device q initial p args).host=.ok schedules) :
    let ready := prefix_ready_full_caller store hm tensor device hs hf hc hd hl ha q initial hp p args route hprofile
    (preparedFamily store ops encoding selection).configure args (launchState tensor device q initial p args)=
      .ok (preparedLaunch store ops encoding args (launchState tensor device q initial p args) ready
        schedules.benchmarks schedules.selected) :=
  preparedFamily_configured store ops encoding selection args _ _ schedules hselect

/-- A selected schedule failure is preserved as an ordinary setup outcome. -/
theorem preparedFamily_selection_failure
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α) (selection : ScheduleSelection Host)
    (args : LaunchArguments) (current : HostState α Host) (error : Failure)
    (h : selection.select args current.host=.error error) :
    (preparedFamily (caller:=caller) store ops encoding selection).configure args current=.error error := by
  simp [preparedFamily,h,Bind.bind,Except.bind]

#axiomsClean preparedFamily_configured
#axiomsClean prefix_family_configured
#axiomsClean preparedFamily_selection_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
