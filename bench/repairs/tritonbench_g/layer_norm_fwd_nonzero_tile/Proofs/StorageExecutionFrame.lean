import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageExecution

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageMachine
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

theorem write_read_frame {α : Type} (c : Config α block) (memory : Memory c)
    (k : AliasMemory.Key) (hk : AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N k)
    (value : Cell α) (l : AliasMemory.Key) (hne : l≠k) :
    TypedPhysical.read (layout c.allocations) (TypedPhysical.write (layout c.allocations) memory k value) l=
      TypedPhysical.read (layout c.allocations) memory l := by
  by_cases hl : (layout c.allocations).live l
  · have ha : (layout c.allocations).address l≠(layout c.allocations).address k :=
      fun he => hne ((c.storage.fresh k hk).2 l hl he)
    simp [TypedPhysical.read,TypedPhysical.write,hl,ha]
  · simp [TypedPhysical.read,hl]

def inputRegion (region : RegionName) : Prop :=
  region∈[KernelEntry.regions.x,KernelEntry.regions.x1,KernelEntry.regions.residual,KernelEntry.regions.weight,
    KernelEntry.regions.bias,KernelEntry.regions.weight1,KernelEntry.regions.bias1,KernelEntry.regions.rowscale,KernelEntry.regions.seeds]

theorem input_not_destination (destination : Compiler.Destination) (region : RegionName) (h : inputRegion region) :
    region≠Address.region KernelEntry.regions (destinationRole destination) := by
  intro he
  subst region
  cases destination with
  | y _ | y1 _ | residualOut _ | mean | rstd =>
    simp_all [inputRegion,KernelEntry.regions,destinationRole,Address.region]
  | mask second _ => cases second <;> simp_all [inputRegion,KernelEntry.regions,destinationRole,Address.region]

theorem step_input_read {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (instruction : Instruction c) (region : RegionName) (hr : inputRegion region) (offset : Nat) :
    TypedPhysical.read (layout c.allocations) (step c artifact memory instruction) (region,offset)=
      TypedPhysical.read (layout c.allocations) memory (region,offset) := by
  rw [step_exact]
  cases he : artifact.stores instruction.2 with
  | none => simp [event,he]
  | some operand =>
    simp only [event,he,Option.map_some]
    apply write_read_frame c memory _ (store_written c artifact instruction operand he)
    intro hkey
    exact input_not_destination instruction.2 region hr (congrArg Prod.fst hkey)

/-- Every graph load still reads the original input cell after a store,
including physically aliased input views. This is derived from output
freshness, not a kernel/input-invariance assumption. -/
theorem step_snapshot {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (instruction : Instruction c) : snapshot c (step c artifact memory instruction)=snapshot c memory := by
  unfold snapshot
  congr 1
  all_goals funext
  all_goals rw [step_input_read c artifact memory instruction _ (by simp [inputRegion])]

theorem event_written {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (inputs : NativeRow.Inputs α)
    (instruction : Instruction c) (e : TypedPhysical.Event α) (he : event c artifact inputs instruction=some e) :
    AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N e.key := by
  obtain ⟨operand,ho,heq⟩ := Option.map_eq_some_iff.mp he
  subst e
  exact store_written c artifact instruction operand ho

/-- The dynamic current-memory evaluator equals its ordered instruction trace
because each preceding store preserves all original input loads. The trace
values are derived instruction-by-instruction, never read from a final output. -/
theorem run_trace {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instructions : List (Instruction c))
    (memory : Memory c) :
    run c artifact instructions memory=TypedPhysical.runPhysical (layout c.allocations) memory
      (instructions.filterMap (event c artifact (snapshot c memory))) := by
  induction instructions generalizing memory with
  | nil => rfl
  | cons instruction rest ih =>
    rw [run,ih,step_snapshot]
    cases he : event c artifact (snapshot c memory) instruction with
    | none => rw [List.filterMap_cons,he]; simpa [step_exact,he] using rfl
    | some e =>
      rw [List.filterMap_cons,he]
      simp only [TypedPhysical.runPhysical,List.foldl_cons]
      rw [step_exact,he]

theorem run_input_read {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instructions : List (Instruction c))
    (memory : Memory c) (region : RegionName) (hr : inputRegion region) (offset : Nat) :
    TypedPhysical.read (layout c.allocations) (run c artifact instructions memory) (region,offset)=
      TypedPhysical.read (layout c.allocations) memory (region,offset) := by
  induction instructions generalizing memory with
  | nil => rfl
  | cons instruction rest ih =>
    rw [run,ih,step_input_read c artifact memory instruction region hr]

theorem run_snapshot {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instructions : List (Instruction c))
    (memory : Memory c) : snapshot c (run c artifact instructions memory)=snapshot c memory := by
  induction instructions generalizing memory with
  | nil => rfl
  | cons instruction rest ih => rw [run,ih,step_snapshot]

theorem run_frame {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instructions : List (Instruction c))
    (memory : Memory c) (i : Fin (layout c.allocations).capacity)
    (h : ∀ k, AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N k →
      (layout c.allocations).address k≠i.val) : run c artifact instructions memory i=memory i := by
  rw [run_trace]
  apply TypedPhysical.trace_frame
  intro e he
  obtain ⟨instruction,_,he⟩ := List.mem_filterMap.mp he
  exact Ne.symm (h e.key (event_written c artifact (snapshot c memory) instruction e he))

#axiomsClean write_read_frame
#axiomsClean input_not_destination
#axiomsClean step_input_read
#axiomsClean step_snapshot
#axiomsClean event_written
#axiomsClean run_trace
#axiomsClean run_input_read
#axiomsClean run_snapshot
#axiomsClean run_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageMachine
