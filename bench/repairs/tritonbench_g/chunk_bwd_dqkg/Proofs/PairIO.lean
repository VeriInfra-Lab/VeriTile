import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.PairIO
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 40000

/-- Same complete source launch, runtime scale word and arbitrary native
routing; only the mathematically equal integer gate bound changes. -/
theorem invocation_run (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (pid : CTA c) (s : State Node) :
    (IO.nativeJob isa route scale c pid).run s=(invocation isa route scale c pid).run s := by
  exact (PairConnection.source_pair
    (FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel)) (nativeConfig c)).run _

theorem invocation_safe (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (pid : CTA c) (bounds : RegionName → Nat) (s : State Node) :
    (IO.nativeJob isa route scale c pid).Safe bounds s ↔ (invocation isa route scale c pid).Safe bounds s := by
  exact (PairConnection.source_pair
    (FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel)) (nativeConfig c)).safe bounds _

theorem schedule_run (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (work : List (CTA c)) (s : State Node) :
    runSchedule (work.map (IO.nativeJob isa route scale c)) s=
      runSchedule (work.map (invocation isa route scale c)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (invocation isa route scale c pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (work : List (CTA c)) (bounds : RegionName → Nat) (s : State Node) :
    ScheduleSafe bounds (work.map (IO.nativeJob isa route scale c)) s ↔
      ScheduleSafe bounds (work.map (invocation isa route scale c)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Every complete current-memory CTA permutation, arbitrary compound Node
inputs, successful safe executions, exact tagged outputs and both raw frames. -/
theorem equivalent (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) (legal : order.Perm (allCTAs c)) :
    (IO.nativeIO isa route scale c order).Equivalent (candidate isa route scale c order) := by
  intro vals _ bounds hb initial hin
  obtain ⟨safe,final,ran,values,frame⟩ :=
    NativeObservation.implements isa route scale c order legal vals trivial bounds hb initial hin
  refine ⟨safe,(schedule_safe isa route scale c order bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run isa route scale c order initial).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.PairIO
