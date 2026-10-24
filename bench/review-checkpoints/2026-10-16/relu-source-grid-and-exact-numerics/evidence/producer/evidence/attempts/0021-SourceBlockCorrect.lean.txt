import bench.tritonbench_g.relu_triton_kernel.Proofs.CheckedBlock

/-! A source-statement trace coupled to checked input/output bytes for the nine
integer/bool profiles. This is a one-program composition, valid for every pid.
It retains the frozen typed AST; native Python/compiler/host correspondence,
other dtype routes and instruction interleaving remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.SourceBlockCorrect
open VeriTile.Triton FP.Structural IntegerWords FiniteAddress TypedValue SourceStages
open AccessMemory CheckedBlock
noncomputable section

/-- Independent byte reference for the original pid-zero write prefix. -/
def byteSpec (M : Algebra (BitVec 32)) (e : Encoding)
    (inputBase outputBase N pid : Nat) (initial : ByteMemory.Memory) : ByteMemory.Memory :=
  if pid = 0 then
    ByteMemory.prefixSpec (ByteMemory.values M e inputBase initial) outputBase initial (min N blockSize)
  else initial

theorem byte_program_spec (M : Algebra (BitVec 32)) (e : Encoding)
    (inputBase outputBase N pid : Nat) (initial : ByteMemory.Memory) :
    ByteMemory.byteProgram M e inputBase outputBase N pid initial =
      byteSpec M e inputBase outputBase N pid initial := by
  by_cases first : pid = 0
  · subst pid
    simp only [ByteMemory.byteProgram, byteSpec, if_true, Nat.zero_mul, Nat.zero_add,
      ByteMemory.prefix_correct]
    rfl
  · simp only [ByteMemory.byteProgram, byteSpec, first, if_false]

theorem byte_spec_frame (M : Algebra (BitVec 32)) (e : Encoding)
    (inputBase outputBase N pid address : Nat) (initial : ByteMemory.Memory)
    (outside : address < outputBase ∨ outputBase + min N blockSize * 4 ≤ address) :
    byteSpec M e inputBase outputBase N pid initial address = initial address := by
  unfold byteSpec
  split
  · rw [← ByteMemory.prefix_correct]
    exact ByteMemory.prefix_frame _ _ _ _ _ outside
  · rfl

specification relu_checked_source_block_correctness
    (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (e : Encoding) (x y : RegionName) (inputBase outputBase N : Nat)
    (initial : State (BitVec 32)) (access : Memory)
    (view : InputWords e (ByteMemory.loadInput e access.bytes inputBase) x N initial.mem)
    (inputReadable : Readable access inputBase (N * elementBytes e))
    (outputWritable : Writable access outputBase (min N blockSize * 4)) :
    ∃ trace final,
      CheckedBlock.program M cmp16 cmp32 e x y inputBase outputBase N initial access = .ok (trace, final) ∧
      Valid (backend M cmp16 cmp32) e x y N blockSize initial trace ∧
      FP.Structural.exec (backend M cmp16 cmp32) (TypedValue.source e.profile x y N blockSize) initial =
        some trace.final ∧
      trace = expected M e x y N blockSize initial ∧
      trace.loaded.regs e.kind.inputType [blockSize] "x" = some (loadedValues e x N blockSize initial) ∧
      trace.result.regs .fp32 [blockSize] "result" = some (resultValues M e x N blockSize initial) ∧
      (∀ i : TileIndex [blockSize], N ≤ offsetValues blockSize initial i →
        loadedValues e x N blockSize initial i = initial.undef e.kind.inputType x (offsetValues blockSize initial i) ∧
        resultValues M e x N blockSize initial i = scalar M e
          (initial.undef e.kind.inputType x (offsetValues blockSize initial i))) ∧
      (∀ i : TileIndex [blockSize], offsetValues blockSize initial i < N →
        resultValues M e x N blockSize initial i = wordValue M e
          (ByteMemory.loadInput e access.bytes inputBase (offsetValues blockSize initial i))) ∧
      final.bytes = byteSpec M e inputBase outputBase N (initial.pids 0) access.bytes ∧
      SameAccess access final ∧
      (∀ address, address < outputBase ∨ outputBase + min N blockSize * 4 ≤ address →
        final.bytes address = access.bytes address ∧ partialContents final address = partialContents access address) := by
  let trace := expected M e x y N blockSize initial
  let final := withBytes access (ByteMemory.byteProgram M e inputBase outputBase N (initial.pids 0) access.bytes)
  have valid := source_trace M cmp16 cmp32 e x y N blockSize initial
  refine ⟨trace, final,
    CheckedBlock.program_success M cmp16 cmp32 e x y inputBase outputBase N initial access view
      inputReadable outputWritable,
    valid, valid_exec _ e x y N blockSize initial trace valid, rfl,
    State.setReg_same _ _ _ _ _, result_register M e x N blockSize initial, ?_,
    active_result M e x inputBase N initial access view,
    byte_program_spec M e inputBase outputBase N (initial.pids 0) access.bytes,
    with_bytes_access access _, ?_⟩
  · intro i inactive
    have unchanged := inactive_load e x N blockSize initial i inactive
    refine ⟨unchanged, ?_⟩
    unfold resultValues
    rw [unchanged]
  · intro address outside
    have unchanged : final.bytes address = access.bytes address := by
      change ByteMemory.byteProgram M e inputBase outputBase N (initial.pids 0) access.bytes address = _
      rw [byte_program_spec]
      exact byte_spec_frame M e inputBase outputBase N (initial.pids 0) address access.bytes outside
    refine ⟨unchanged, ?_⟩
    change (if access.allocated address = Bool.true then some (final.bytes address) else none) = _
    rw [unchanged]
    rfl

#specNonCircular byteSpec avoiding [CheckedBlock.program, checkedLoad, checkedGate,
  TypedValue.source, materialized, FP.Structural.exec, step, CheckedTileIO.storeMapped,
  CheckedTileIO.loadMapped, ByteMemory.byteProgram]
#axiomsClean byte_program_spec
#axiomsClean byte_spec_frame
#axiomsClean relu_checked_source_block_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.SourceBlockCorrect
