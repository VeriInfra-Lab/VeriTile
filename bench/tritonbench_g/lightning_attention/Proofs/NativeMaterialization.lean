import bench.tritonbench_g.lightning_attention.Proofs.NativeCoverage

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeMaterialization
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open NativeLocality NativeCoverage
noncomputable section
set_option maxHeartbeats 6000000

def StageCovered (scope : RegionName → Nat → Prop) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) : Prop :=
  match stage with
  | .forward => ForwardCovered scope d.parameters pid
  | .intra | .inter => HeadCovered scope d.parameters pid.head

theorem stage_agree (scope : RegionName → Nat → Prop) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (covered : StageCovered scope stage d pid)
    (a b : Memory Native.Expr) (h : Agree scope a b) :
    Agree scope (stageMemory Native.algebra Native.dots stage d pid a)
      (stageMemory Native.algebra Native.dots stage d pid b) := by
  cases stage with
  | forward => exact fwdMemory_agree _ _ _ _ _ covered a b a b h h
  | intra => exact intraMemory_agree _ _ _ _ _ _ covered a b h
  | inter => exact interMemory_agree _ _ _ _ _ covered a b a b h h

theorem initial_eq (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (memory : Memory Word) :
    (NativeIO.launch ops plan stage d pid).source.initial
      ((NativeIO.launch ops plan stage d pid).embedded memory) =
    launchState stage d pid ((NativeIO.launch ops plan stage d pid).embed memory) := rfl

theorem embed_agree (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (scope : RegionName → Nat → Prop)
    (memory : Memory Word) (words : Native.WordMemory) (h : Agree scope memory (typed words)) :
    Agree scope ((NativeIO.launch ops plan stage d pid).embed memory) (Native.embed words) := by
  intro r o hr
  change mapCell Native.Expr.bits (memory r o) = .mk .real (.bits (words r o))
  rw [h r o hr]
  rfl

theorem finish_typed (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (memory : Memory Native.Expr) (h : AllReal memory) :
    (NativeIO.launch ops plan stage d pid).finish memory = typed (Native.observe ops plan memory) := by
  funext r o
  obtain ⟨v,hv⟩ := h r o
  change mapCell (fun x => (Native.lower plan x).eval ops) (memory r o) =
    .mk .real ((Native.lower plan ((memory r o).read .real)).eval ops)
  rw [hv]
  rfl

theorem run_model (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (memory : Memory Word) :
    (NativeIO.launch ops plan stage d pid).run memory =
      some ((NativeIO.launch ops plan stage d pid).finish
        (stageMemory Native.algebra Native.dots stage d pid
          ((NativeIO.launch ops plan stage d pid).embed memory))) := by
  obtain ⟨t,ran,hm⟩ := stageProgram Native.algebra nativeMasks Native.dots stage d pid
    ((NativeIO.launch ops plan stage d pid).embed memory)
  have jobRun : (NativeIO.launch ops plan stage d pid).source.run
      ((NativeIO.launch ops plan stage d pid).embedded memory) = some t := by
    simpa only [Invocation.run,initial_eq,NativeIO.launch,invocation] using ran
  rw [←hm]
  exact MaterializedInvocation.run_eq _ _ t jobRun

/-- Exact connection to the old arbitrary-word execution model, including
every exterior cell and materialization before the following CTA. -/
theorem run_typed (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (words : Native.WordMemory) :
    (NativeIO.launch ops plan stage d pid).run (typed words) =
      some (typed (Native.stageReference ops plan stage d pid words)) := by
  rw [run_model]
  have real := stageMemory_real Native.algebra Native.dots stage d pid (Native.embed words)
    (fun r o => ⟨.bits (words r o),rfl⟩)
  change some ((NativeIO.launch ops plan stage d pid).finish
    (stageMemory Native.algebra Native.dots stage d pid (Native.embed words))) = _
  rw [finish_typed _ _ _ _ _ _ real]
  rfl

theorem stage_result (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (scope : RegionName → Nat → Prop)
    (covered : StageCovered scope stage d pid)
    (memory : Memory Word) (words : Native.WordMemory) (h : Agree scope memory (typed words)) :
    ∃ final, (NativeIO.launch ops plan stage d pid).run memory = some final ∧
      Agree scope final (typed (Native.stageReference ops plan stage d pid words)) ∧
      (∀ r o, ¬Footprint stage d pid r o → final r o = memory r o) := by
  let job := NativeIO.launch ops plan stage d pid
  let actual := stageMemory Native.algebra Native.dots stage d pid (job.embed memory)
  let expected := stageMemory Native.algebra Native.dots stage d pid (Native.embed words)
  have agrees : Agree scope actual expected :=
    stage_agree scope stage d pid covered _ _ (embed_agree ops plan stage d pid scope memory words h)
  have real : AllReal expected :=
    stageMemory_real _ _ _ _ _ _ (fun r o => ⟨.bits (words r o),rfl⟩)
  refine ⟨job.finish actual,run_model _ _ _ _ _ _,?_,?_⟩
  · intro r o hr
    have he := agrees r o hr
    have ht := congrFun (congrFun (finish_typed ops plan stage d pid expected real) r) o
    change mapCell _ (actual r o) = _
    rw [he]
    exact ht
  · intro r o outside
    apply MaterializedInvocation.finish_frame job memory actual r o
    exact stage_frame Native.algebra Native.dots stage d pid (job.embed memory) r o outside

theorem grid_result (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pids : List PID) (scope : RegionName → Nat → Prop)
    (covered : ∀ pid∈pids, StageCovered scope stage d pid)
    (memory : Memory Word) (words : Native.WordMemory) (h : Agree scope memory (typed words)) :
    ∃ final, runMaterializedSchedule (pids.map (NativeIO.launch ops plan stage d)) memory = some final ∧
      Agree scope final (typed (Native.gridReference ops plan stage d pids words)) ∧
      (∀ r o, (∀ pid∈pids, ¬Footprint stage d pid r o) → final r o = memory r o) := by
  induction pids generalizing memory words with
  | nil => exact ⟨memory,rfl,h,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    obtain ⟨next,ran,hn,fn⟩ := stage_result ops plan stage d pid scope
      (covered pid (by simp)) memory words h
    obtain ⟨final,restRan,hf,ff⟩ := ih
      (fun id hi => covered id (by simp [hi])) next
      (Native.stageReference ops plan stage d pid words) hn
    refine ⟨final,?_,hf,?_⟩
    · simp only [List.map_cons,runMaterializedSchedule,ran,Option.bind_some,restRan]
    · intro r o outside
      rw [ff r o (fun id hi => outside id (by simp [hi]))]
      exact fn r o (outside pid (by simp))

theorem grid_typed (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pids : List PID) (words : Native.WordMemory) :
    runMaterializedSchedule (pids.map (NativeIO.launch ops plan stage d)) (typed words) =
      some (typed (Native.gridReference ops plan stage d pids words)) := by
  induction pids generalizing words with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runMaterializedSchedule,run_typed,Option.bind_some]
    exact ih _

#axiomsClean run_typed
#axiomsClean grid_typed
#axiomsClean grid_result
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeMaterialization

