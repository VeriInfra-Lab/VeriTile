import bench.repairs.tritonbench_g.rotary_transform_tile_checked.PairIO
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.PairIO
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem invocation_run (p : Ops α) (c : Config) (cu64 off64 : Bool) (pid : Program) (s : State α) :
    (invocation p c cu64 off64 pid).run s = (PublicIO.invocation p c cu64 off64 pid).run s :=
  (IndexCSE.execution (KernelAlgebra.backend p) c.shape cu64 off64
    ((PublicIO.invocation p c cu64 off64 pid).initial s)).symm

theorem invocation_safe (p : Ops α) (c : Config) (cu64 off64 : Bool) (pid : Program)
    (bounds : RegionName → Nat) (s : State α) :
    (PublicIO.invocation p c cu64 off64 pid).Safe bounds s ↔
      (invocation p c cu64 off64 pid).Safe bounds s :=
  IndexCSE.trace_safe (KernelAlgebra.backend p) c.shape cu64 off64 bounds
    ((PublicIO.invocation p c cu64 off64 pid).initial s)

theorem schedule_run (p : Ops α) (c : Config) (cu64 off64 : Bool)
    (jobs : List Program) (s : State α) :
    runSchedule (jobs.map (invocation p c cu64 off64)) s =
      runSchedule (jobs.map (PublicIO.invocation p c cu64 off64)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    exact congrArg (fun f => (PublicIO.invocation p c cu64 off64 pid).run s >>= f) (funext ih)

theorem schedule_safe (p : Ops α) (c : Config) (cu64 off64 : Bool) (jobs : List Program)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (PublicIO.invocation p c cu64 off64)) s ↔
      ScheduleSafe bounds (jobs.map (invocation p c cu64 off64)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_run]
    exact and_congr (invocation_safe p c cu64 off64 pid bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

theorem full_run (p : Ops α) (observe : α → α) (c : Config) (cu64 off64 : Bool)
    (xs : ChannelValues α (PublicIO.inputs c)) (s : State α) :
    runSchedule ((candidate p observe c cu64 off64).schedule xs) s =
      runSchedule ((PublicIO.io p observe c cu64 off64).schedule xs) s := by
  by_cases empty : c.shape.total=0 ∨ c.shape.half=0
  · simp only [candidate,PublicIO.io,if_pos empty]
  · simpa only [candidate,PublicIO.io,if_neg empty] using schedule_run p c cu64 off64 (programs c) s

theorem full_safe (p : Ops α) (observe : α → α) (c : Config) (cu64 off64 : Bool)
    (xs : ChannelValues α (PublicIO.inputs c)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds ((PublicIO.io p observe c cu64 off64).schedule xs) s ↔
      ScheduleSafe bounds ((candidate p observe c cu64 off64).schedule xs) s := by
  by_cases empty : c.shape.total=0 ∨ c.shape.half=0
  · simp only [candidate,PublicIO.io,if_pos empty]
  · simpa only [candidate,PublicIO.io,if_neg empty] using schedule_safe p c cu64 off64 (programs c) bounds s

/-- The whole candidate grid on the exact original native/logical domain:
every active result read (including suffix), both metadata widths, all flags,
actual safe successful execution and the unchanged exterior raw-cell frame. -/
theorem candidate_native (c : Config) (valid : c.Valid) (p : Native.Instructions) (cu64 off64 : Bool) :
    (candidate (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).toReadProgramIO.ImplementsOn
      (PublicIO.domain c)
      (PublicIO.expected c (fun values b r h d =>
        Native.atom (nativeFormula c p (Native.project values) b r h d))) := by
  intro xs legal bounds backed initial inputs
  obtain ⟨safe,final,ran,outputs,frame⟩ := IOResults.native c valid p cu64 off64
    xs legal bounds backed initial inputs
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (full_safe (Native.ops p) (fun node => Native.atom node.value) c cu64 off64 xs bounds initial).mp safe
  · exact (full_run (Native.ops p) (fun node => Native.atom node.value) c cu64 off64 xs initial).trans ran

theorem equivalent (c : Config) (valid : c.Valid) (p : Native.Instructions) (cu64 off64 : Bool) :
    (PublicIO.io (Native.ops p) (fun node => Native.atom node.value) c cu64 off64).EquivalentOn
      (candidate (Native.ops p) (fun node => Native.atom node.value) c cu64 off64) (PublicIO.domain c) :=
  ReadProgramIO.EquivalentOn.of_implementsOn (IOResults.native c valid p cu64 off64)
    (candidate_native c valid p cu64 off64)

#axiomsClean candidate_native
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.PairIO
