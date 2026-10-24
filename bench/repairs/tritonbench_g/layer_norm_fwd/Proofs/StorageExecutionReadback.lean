import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageExecutionFrame

namespace VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

theorem store_written_at {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instruction : Instruction c)
    (operand : StorageCompiler.Operand) (h : artifact.stores instruction.2=some operand) :
    Contract.written c.flags preparedRegions c.strides c.M c.N instruction.1.val
      (key c instruction).1 (key c instruction).2 := by
  obtain ⟨ha,hl⟩ := store_enabled c artifact instruction.2 operand h
  obtain ⟨row,destination⟩ := instruction
  cases destination with
  | y _ | y1 _ | residualOut _ | mean | rstd =>
    simp_all [key,destinationRole,destinationCol,active,NativeAccess.laneActive,region,offset,Contract.written,Contract.inRow_of_col]
  | mask second _ =>
    cases second <;> simp_all [key,destinationRole,destinationCol,active,NativeAccess.laneActive,region,offset,Contract.written,Contract.inRow_of_col]

theorem key_injective {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (left right : Instruction c) (a b : StorageCompiler.Operand)
    (ha : artifact.stores left.2=some a) (hb : artifact.stores right.2=some b)
    (he : key c left=key c right) : left=right := by
  have hw := store_written_at c artifact right b hb
  rw [←he] at hw
  have hr := NativeGeometry.written_row_unique c.flags preparedRegions c.strides c.M c.N
    left.1.val right.1.val c.geometry left.1.isLt right.1.isLt _ _
    (store_written_at c artifact left a ha) hw
  have hfin : left.1=right.1 := Fin.ext hr
  obtain ⟨row,destination⟩ := left
  obtain ⟨other,target⟩ := right
  dsimp only at hfin
  subst other
  have hl := (store_enabled c artifact destination a ha).2
  have ht := (store_enabled c artifact target b hb).2
  have hM : 0<c.M := lt_of_le_of_lt (Nat.zero_le row.val) row.isLt
  have hmul : c.N≤c.M*c.N := Nat.le_mul_of_pos_left _ hM
  clear ha hb hw hr
  congr 1
  cases destination <;> cases target
  all_goals simp only [key,destinationRole,destinationCol,region,offset,preparedRegions,
    NativeAccess.laneActive,Prod.mk.injEq] at he hl ht
  all_goals try split_ifs at he hl ht
  all_goals simp_all [region,offset,NativeAccess.laneActive,Nat.add_mul]
  all_goals first | (congr 1 <;> omega) | omega

theorem write_read_same {α : Type} (c : Config α block) (memory : Memory c)
    (k : AliasMemory.Key) (hk : AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N k)
    (value : Cell α) :
    TypedPhysical.read (layout c.allocations) (TypedPhysical.write (layout c.allocations) memory k value) k=value := by
  simp [TypedPhysical.read,TypedPhysical.write,(c.storage.fresh k hk).1]

theorem trace_preserves_value {α : Type} (c : Config α block) (events : List (TypedPhysical.Event α))
    (hw : ∀ e∈events, AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N e.key)
    (k : AliasMemory.Key) (value : Cell α)
    (hv : ∀ e∈events, e.key=k → e.value=value) (memory : Memory c)
    (hread : TypedPhysical.read (layout c.allocations) memory k=value) :
    TypedPhysical.read (layout c.allocations) (TypedPhysical.runPhysical (layout c.allocations) memory events) k=value := by
  induction events generalizing memory with
  | nil => exact hread
  | cons e es ih =>
    apply ih (fun e he => hw e (List.mem_cons_of_mem _ he))
      (fun e he => hv e (List.mem_cons_of_mem _ he))
    by_cases he : k=e.key
    · rw [he,write_read_same c memory _ (hw e (List.mem_cons_self ..))]
      exact hv e (List.mem_cons_self ..) he.symm
    · exact (write_read_frame c memory _ (hw e (List.mem_cons_self ..)) e.value k he).trans hread

theorem trace_readback {α : Type} (c : Config α block) (events : List (TypedPhysical.Event α))
    (hw : ∀ e∈events, AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N e.key)
    (e : TypedPhysical.Event α) (he : e∈events)
    (hv : ∀ other∈events, other.key=e.key → other.value=e.value) (memory : Memory c) :
    TypedPhysical.read (layout c.allocations) (TypedPhysical.runPhysical (layout c.allocations) memory events) e.key=e.value := by
  induction events generalizing memory with
  | nil => simp at he
  | cons first rest ih =>
    rcases List.mem_cons.mp he with he | he
    · subst first
      apply trace_preserves_value c rest (fun e he => hw e (List.mem_cons_of_mem _ he)) e.key e.value
        (fun e he => hv e (List.mem_cons_of_mem _ he))
      exact write_read_same c memory _ (hw e (List.mem_cons_self ..)) e.value
    · exact ih (fun e he => hw e (List.mem_cons_of_mem _ he)) he
        (fun e he => hv e (List.mem_cons_of_mem _ he)) _

theorem event_key_value {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (inputs : NativeRow.Inputs α)
    (left right : Instruction c) (a b : TypedPhysical.Event α)
    (ha : event c artifact inputs left=some a) (hb : event c artifact inputs right=some b)
    (he : a.key=b.key) : a=b := by
  obtain ⟨operand,ho,hae⟩ := Option.map_eq_some_iff.mp ha
  obtain ⟨other,ht,hbe⟩ := Option.map_eq_some_iff.mp hb
  subst a
  subst b
  have hi := key_injective c artifact left right operand other ho ht he
  subst right
  have hop : operand=other := Option.some.inj (ho.symm.trans ht)
  subst other
  rfl

theorem program_contains {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instruction : Instruction c)
    (operand : StorageCompiler.Operand) (he : artifact.stores instruction.2=some operand) :
    instruction∈program c := by
  have hc := store_column c instruction.2 (store_enabled c artifact instruction.2 operand he).2
  obtain ⟨row,destination⟩ := instruction
  apply List.mem_flatMap.mpr
  refine ⟨row,by simp,?_⟩
  apply List.mem_map.mpr
  refine ⟨destination,?_,rfl⟩
  cases destination with
  | y col | y1 col | residualOut col | mean | rstd => simp_all [rowDestinations,destinationCol]
  | mask second col => cases second <;> simp_all [rowDestinations,destinationCol]

/-- Every enabled source destination reads the converted value of its actual
emitted operand after the complete current-memory execution. -/
theorem execute_readback {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (instruction : Instruction c) (operand : StorageCompiler.Operand)
    (he : artifact.stores instruction.2=some operand) :
    TypedPhysical.read (layout c.allocations) (execute c artifact memory) (key c instruction)=
      operand.eval c.ops c.casts (snapshot c memory) c.M instruction.1.val c.eps c.p := by
  unfold execute
  rw [run_trace]
  let e : TypedPhysical.Event α :=
    ⟨key c instruction,operand.eval c.ops c.casts (snapshot c memory) c.M instruction.1.val c.eps c.p⟩
  have hevent : event c artifact (snapshot c memory) instruction=some e := by simp [event,he,e]
  refine trace_readback c _ ?_ e ?_ ?_ memory
  · intro other hm
    obtain ⟨ins,_,hh⟩ := List.mem_filterMap.mp hm
    exact event_written c artifact _ ins other hh
  · exact List.mem_filterMap.mpr ⟨instruction,program_contains c artifact instruction operand he,hevent⟩
  · intro other hm hkey
    obtain ⟨ins,_,hh⟩ := List.mem_filterMap.mp hm
    exact congrArg TypedPhysical.Event.value (event_key_value c artifact _ ins instruction other e hh hevent hkey)

#axiomsClean store_written_at
#axiomsClean key_injective
#axiomsClean write_read_same
#axiomsClean trace_preserves_value
#axiomsClean trace_readback
#axiomsClean event_key_value
#axiomsClean program_contains
#axiomsClean execute_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
