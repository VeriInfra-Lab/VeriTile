import bench.tritonbench_g.relu_triton_kernel.SourceBlockCorrect

/-! Complete schedules of the source-coupled checked program. Each program
starts from the preceding program's logical state and byte memory. A trace is
retained for every pid, including its pre-gate load and result statement.
Schedules are permutations of whole programs, not native instruction traces. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.CheckedSchedule
open VeriTile.Triton FP.Structural IntegerWords FiniteAddress TypedValue SourceStages
open AccessMemory CheckedBlock
noncomputable section

def runSchedule (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (e : Encoding) (x y : RegionName) (inputBase outputBase N : Nat) :
    List Nat → State (BitVec 32) → Memory → Result (List (Trace (BitVec 32)) × State (BitVec 32) × Memory)
  | [], logical, mem => .ok ([], logical, mem)
  | pid :: rest, logical, mem => do
    let (trace, bytes) ← CheckedBlock.program M cmp16 cmp32 e x y inputBase outputBase N
      (GatedGrid.atPid logical pid) mem
    let (traces, finalLogical, finalBytes) ← runSchedule M cmp16 cmp32 e x y inputBase outputBase N
      rest trace.final bytes
    pure (trace :: traces, finalLogical, finalBytes)

/-- Independent scalar/register schedule reference, without source execution
or checked-byte routines. It retains all seven states for every pid. -/
def scheduleSpec (M : Algebra (BitVec 32)) (e : Encoding) (x y : RegionName) (N : Nat) :
    List Nat → State (BitVec 32) → List (Trace (BitVec 32)) × State (BitVec 32)
  | [], initial => ([], initial)
  | pid :: rest, initial =>
    let trace := expected M e x y N blockSize (GatedGrid.atPid initial pid)
    let tail := scheduleSpec M e x y N rest trace.final
    (trace :: tail.1, tail.2)

def SourceSchedule (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (e : Encoding) (x y : RegionName) (N : Nat) :
    List Nat → State (BitVec 32) → List (Trace (BitVec 32)) → State (BitVec 32) → Prop
  | [], initial, [], final => final = initial
  | pid :: rest, initial, trace :: traces, final =>
    Valid (backend M cmp16 cmp32) e x y N blockSize (GatedGrid.atPid initial pid) trace ∧
    FP.Structural.exec (backend M cmp16 cmp32) (TypedValue.source e.profile x y N blockSize)
      (GatedGrid.atPid initial pid) = some trace.final ∧
    trace.result.regs .fp32 [blockSize] "result" =
      some (resultValues M e x N blockSize (GatedGrid.atPid initial pid)) ∧
    SourceSchedule M cmp16 cmp32 e x y N rest trace.final traces final
  | _, _, _, _ => False

theorem schedule_spec_trace (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N : Nat) (order : List Nat) (initial : State (BitVec 32)) :
    SourceSchedule M cmp16 cmp32 e x y N order initial
      (scheduleSpec M e x y N order initial).1 (scheduleSpec M e x y N order initial).2 := by
  induction order generalizing initial with
  | nil => rfl
  | cons pid rest ih =>
    have valid := source_trace M cmp16 cmp32 e x y N blockSize (GatedGrid.atPid initial pid)
    exact ⟨valid, valid_exec _ _ _ _ _ _ _ _ valid,
      result_register M e x N blockSize (GatedGrid.atPid initial pid), ih _⟩

theorem schedule_spec_length (M : Algebra (BitVec 32)) (e : Encoding)
    (x y : RegionName) (N : Nat) (order : List Nat) (initial : State (BitVec 32)) :
    (scheduleSpec M e x y N order initial).1.length = order.length := by
  induction order generalizing initial with
  | nil => rfl
  | cons pid rest ih => exact congrArg Nat.succ (ih _)

theorem block_memory (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N pid : Nat) (initial : State (BitVec 32)) :
    (expected M e x y N blockSize (GatedGrid.atPid initial pid)).final.mem =
      if pid = 0 then reference M cmp16 cmp32 e.profile.kind x y N blockSize initial.mem else initial.mem := by
  have execution := valid_exec (backend M cmp16 cmp32) e x y N blockSize
    (GatedGrid.atPid initial pid) _ (source_trace M cmp16 cmp32 e x y N blockSize (GatedGrid.atPid initial pid))
  obtain ⟨final, ran, memory⟩ := pid_memory M cmp16 cmp32 e.profile.kind x y N blockSize initial pid
  have same := Option.some.inj (ran.symm.trans execution)
  rw [same] at memory
  exact memory

theorem block_input (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N pid index : Nat) (initial : State (BitVec 32)) (distinct : x ≠ y) :
    (expected M e x y N blockSize (GatedGrid.atPid initial pid)).final.mem x index = initial.mem x index := by
  rw [block_memory M cmp16 cmp32]
  split
  · exact reference_frame M cmp16 cmp32 e.profile.kind x y N blockSize initial.mem x index (Or.inl distinct)
  · rfl

theorem byte_program_input (M : Algebra (BitVec 32)) (e : Encoding)
    (inputBase outputBase N pid index : Nat) (initial : ByteMemory.Memory)
    (separate : ByteMemory.Separate e inputBase outputBase N) (active : index < N) :
    ByteMemory.loadInput e (ByteMemory.byteProgram M e inputBase outputBase N pid initial) inputBase index =
      ByteMemory.loadInput e initial inputBase index := by
  by_cases first : pid = 0
  · subst pid
    simp only [ByteMemory.byteProgram, if_true, Nat.zero_mul, Nat.zero_add]
    exact ByteMemory.prefix_input_preserved e _ inputBase outputBase N _ index initial separate
      (Nat.min_le_left _ _) active
  · simp only [ByteMemory.byteProgram, first, if_false]

theorem next_input_view (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding) (x y : RegionName)
    (inputBase outputBase N pid : Nat) (initial : State (BitVec 32)) (mem : Memory)
    (view : InputWords e (ByteMemory.loadInput e mem.bytes inputBase) x N initial.mem)
    (distinct : x ≠ y) (separate : ByteMemory.Separate e inputBase outputBase N) :
    InputWords e (ByteMemory.loadInput e
      (ByteMemory.byteProgram M e inputBase outputBase N pid mem.bytes) inputBase) x N
      (expected M e x y N blockSize (GatedGrid.atPid initial pid)).final.mem := by
  intro index active
  rw [block_input M cmp16 cmp32 e x y N pid index initial distinct,
    byte_program_input M e inputBase outputBase N pid index mem.bytes separate active]
  exact view index active

theorem schedule_success (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding) (x y : RegionName)
    (inputBase outputBase N : Nat) (order : List Nat) (initial : State (BitVec 32)) (mem : Memory)
    (view : InputWords e (ByteMemory.loadInput e mem.bytes inputBase) x N initial.mem)
    (distinct : x ≠ y) (separate : ByteMemory.Separate e inputBase outputBase N)
    (inputReadable : Readable mem inputBase (N * elementBytes e))
    (outputWritable : Writable mem outputBase (min N blockSize * 4)) :
    runSchedule M cmp16 cmp32 e x y inputBase outputBase N order initial mem =
      .ok ((scheduleSpec M e x y N order initial).1, (scheduleSpec M e x y N order initial).2,
        withBytes mem (AccessMemory.effectSchedule M e inputBase outputBase N order mem.bytes)) := by
  induction order generalizing initial mem with
  | nil => rfl
  | cons pid rest ih =>
    rw [runSchedule, CheckedBlock.program_success M cmp16 cmp32 e x y inputBase outputBase N
      (GatedGrid.atPid initial pid) mem view inputReadable outputWritable]
    change (do
      let (traces, finalLogical, finalBytes) ← runSchedule M cmp16 cmp32 e x y inputBase outputBase N
        rest (expected M e x y N blockSize (GatedGrid.atPid initial pid)).final
        (withBytes mem (ByteMemory.byteProgram M e inputBase outputBase N pid mem.bytes))
      pure (expected M e x y N blockSize (GatedGrid.atPid initial pid) :: traces, finalLogical, finalBytes)) = _
    rw [ih (expected M e x y N blockSize (GatedGrid.atPid initial pid)).final
      (withBytes mem (ByteMemory.byteProgram M e inputBase outputBase N pid mem.bytes))
      (next_input_view M cmp16 cmp32 e x y inputBase outputBase N pid initial mem view distinct separate)
      inputReadable outputWritable]
    rfl

def logicalEffect (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (e : Encoding) (x y : RegionName) (N : Nat) :
    List Nat → GatedGrid.Memory (BitVec 32) → GatedGrid.Memory (BitVec 32)
  | [], mem => mem
  | pid :: rest, mem => logicalEffect M cmp16 cmp32 e x y N rest
      (if pid = 0 then reference M cmp16 cmp32 e.profile.kind x y N blockSize mem else mem)

theorem schedule_memory (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N : Nat) (order : List Nat) (initial : State (BitVec 32)) :
    (scheduleSpec M e x y N order initial).2.mem =
      logicalEffect M cmp16 cmp32 e x y N order initial.mem := by
  induction order generalizing initial with
  | nil => rfl
  | cons pid rest ih =>
    change (scheduleSpec M e x y N rest
      (expected M e x y N blockSize (GatedGrid.atPid initial pid)).final).2.mem = _
    rw [ih, block_memory M cmp16 cmp32]
    rfl

theorem logical_effect_perm (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N : Nat) {left right : List Nat} (perm : left.Perm right)
    (mem : GatedGrid.Memory (BitVec 32)) :
    logicalEffect M cmp16 cmp32 e x y N left mem = logicalEffect M cmp16 cmp32 e x y N right mem := by
  induction perm generalizing mem with
  | nil => rfl
  | cons pid perm ih => exact ih _
  | swap p q rest =>
    by_cases hp : p = 0 <;> by_cases hq : q = 0 <;> simp only [logicalEffect, hp, hq, if_true, if_false]
  | trans first second ih₁ ih₂ => exact (ih₁ mem).trans (ih₂ mem)

theorem logical_effect_append (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N : Nat) (left right : List Nat) (mem : GatedGrid.Memory (BitVec 32)) :
    logicalEffect M cmp16 cmp32 e x y N (left ++ right) mem =
      logicalEffect M cmp16 cmp32 e x y N right (logicalEffect M cmp16 cmp32 e x y N left mem) := by
  induction left generalizing mem with
  | nil => rfl
  | cons pid rest ih => exact ih _

theorem logical_effect_range (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding)
    (x y : RegionName) (N count : Nat) (mem : GatedGrid.Memory (BitVec 32)) :
    logicalEffect M cmp16 cmp32 e x y N (List.range count) mem =
      if count = 0 then mem else reference M cmp16 cmp32 e.profile.kind x y N blockSize mem := by
  induction count with
  | zero => rfl
  | succ count ih =>
    rw [List.range_succ, logical_effect_append, ih]
    cases count with
    | zero => rfl
    | succ count => simp only [logicalEffect, Nat.succ_ne_zero, if_false]

theorem complete_schedule_memory (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding) (x y : RegionName)
    (inputBase N : Nat) (order : List Nat)
    (complete : order.Perm (List.range (GatedGrid.gridCount N blockSize)))
    (initial : State (BitVec 32)) (mem : Memory)
    (view : InputWords e (ByteMemory.loadInput e mem.bytes inputBase) x N initial.mem) :
    (scheduleSpec M e x y N order initial).2.mem =
      wordSpec M e (ByteMemory.loadInput e mem.bytes inputBase) y N blockSize initial.mem := by
  rw [schedule_memory M cmp16 cmp32, logical_effect_perm M cmp16 cmp32 e x y N complete,
    logical_effect_range, reference_refines M cmp16 cmp32 e _ x y N blockSize initial.mem view]
  rfl

theorem complete_schedule_success (M : Algebra (BitVec 32))
    (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool) (e : Encoding) (x y : RegionName)
    (inputBase outputBase N : Nat) (order : List Nat)
    (complete : order.Perm (List.range (GatedGrid.gridCount N blockSize)))
    (initial : State (BitVec 32)) (mem : Memory)
    (view : InputWords e (ByteMemory.loadInput e mem.bytes inputBase) x N initial.mem)
    (distinct : x ≠ y) (separate : ByteMemory.Separate e inputBase outputBase N)
    (inputReadable : Readable mem inputBase (N * elementBytes e))
    (outputWritable : Writable mem outputBase (min N blockSize * 4)) :
    runSchedule M cmp16 cmp32 e x y inputBase outputBase N order initial mem =
      .ok ((scheduleSpec M e x y N order initial).1, (scheduleSpec M e x y N order initial).2,
        withBytes mem (ByteMemory.byteGrid M e inputBase outputBase N mem.bytes)) := by
  rw [schedule_success M cmp16 cmp32 e x y inputBase outputBase N order initial mem view distinct separate
    inputReadable outputWritable, AccessMemory.effect_schedule_perm M e inputBase outputBase N complete,
    AccessMemory.effect_range]
  rfl

#specNonCircular scheduleSpec avoiding [CheckedBlock.program, checkedLoad, checkedGate,
  TypedValue.source, materialized, FP.Structural.exec, step, runSchedule]
#axiomsClean schedule_spec_trace
#axiomsClean schedule_spec_length
#axiomsClean block_memory
#axiomsClean block_input
#axiomsClean byte_program_input
#axiomsClean next_input_view
#axiomsClean schedule_success
#axiomsClean schedule_memory
#axiomsClean logical_effect_perm
#axiomsClean logical_effect_append
#axiomsClean logical_effect_range
#axiomsClean complete_schedule_memory
#axiomsClean complete_schedule_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.CheckedSchedule
