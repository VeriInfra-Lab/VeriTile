import bench.tritonbench_g.relu_triton_kernel.Proofs.AccessGrid

/-! Successful partial mathematical byte execution and complete program orders.
Input readability covers all N words. Output writability covers only the first
min(N,1024) fp32 words; no availability constraint is placed on the output tail.
The separate native storage/compiler/host/error correspondence remains open. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.AccessGridCorrect
open VeriTile.Triton FP.Structural GatedGrid TypedValue IntegerWords FiniteAddress ByteMemory
noncomputable section

specification relu_access_grid_correctness
    (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (encoding : Encoding) (inputBase outputBase N : Nat) (x y : RegionName)
    (access : AccessMemory.Memory) (initial : State (BitVec 32))
    (order : List Nat) (complete : order.Perm (List.range (gridCount N blockSize)))
    (inputReadable : AccessMemory.Readable access inputBase (N * elementBytes encoding))
    (outputWritable : AccessMemory.Writable access outputBase (min N blockSize * 4))
    (view : LogicalView encoding access.bytes inputBase outputBase N x y initial.mem)
    (distinct : x ≠ y) (separate : Separate encoding inputBase outputBase N)
    (indexDomain : IndexDomain N)
    (inputDomain : ByteDomain inputBase (elementBytes encoding) N)
    (outputDomain : ByteDomain outputBase 4 N) :
    ∃ final logicalFinal,
      AccessMemory.runSchedule M encoding inputBase outputBase N order access = .ok final ∧
      AccessMemory.runGrid M encoding inputBase outputBase N access = .ok final ∧
      GatedGrid.runGrid (backend M cmp16 cmp32) (source encoding.profile x y N blockSize)
        N blockSize initial = some logicalFinal ∧
      logicalFinal.mem = wordSpec M encoding (ByteMemory.loadInput encoding access.bytes inputBase)
        y N blockSize initial.mem ∧
      final.bytes = byteGrid M encoding inputBase outputBase N access.bytes ∧
      final.bytes = byteSpec M encoding inputBase outputBase N access.bytes ∧
      LogicalView encoding final.bytes inputBase outputBase N x y logicalFinal.mem ∧
      AccessMemory.SameAccess access final ∧
      (∀ address, address < outputBase ∨ outputBase + min N blockSize * 4 ≤ address →
        final.bytes address = access.bytes address ∧
        AccessMemory.partialContents final address = AccessMemory.partialContents access address) ∧
      (∀ pid, pid < gridCount N blockSize → ∀ lane, lane < blockSize → offset pid lane < N →
        AccessMemory.loadInputAt64 encoding access (pointer64 inputBase (elementBytes encoding) pid lane) =
          .ok (ByteMemory.loadInput encoding access.bytes inputBase (offset pid lane))) ∧
      AddressGridCorrect.AddressCertificate encoding N inputBase outputBase := by
  obtain ⟨logicalFinal, he, hm, hv, hb, hf, _, _, address⟩ :=
    ByteGridCorrect.relu_byte_grid_correctness M cmp16 cmp32 encoding inputBase outputBase N x y access.bytes
      initial view distinct separate indexDomain inputDomain outputDomain
  refine ⟨AccessMemory.withBytes access (byteGrid M encoding inputBase outputBase N access.bytes),
    logicalFinal,
    AccessMemory.complete_schedule_success M encoding inputBase outputBase N order complete access
      inputReadable outputWritable,
    AccessMemory.grid_success M encoding inputBase outputBase N access inputReadable outputWritable,
    he, hm, rfl, hb, hv, AccessMemory.with_bytes_access access _, ?_, ?_, address⟩
  · intro addr outside
    exact ⟨hf addr outside,
      AccessMemory.grid_partial_frame M encoding inputBase outputBase N addr access outside⟩
  · intro pid launched lane inBlock active
    rw [AccessMemory.finite_input_access encoding access inputBase N pid lane indexDomain inputDomain
      launched inBlock active]
    exact AccessMemory.load_input_success encoding access inputBase (offset pid lane)
      (AccessMemory.readable_element encoding access inputBase N (offset pid lane) inputReadable active)

/-- The latent total payload always admits the old logical view. This does not
assert tail readability or native allocation; actual accesses use permissions. -/
theorem constructed_access_grid
    (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (encoding : Encoding) (inputBase outputBase N : Nat) (x y : RegionName)
    (access : AccessMemory.Memory) (initial : State (BitVec 32))
    (order : List Nat) (complete : order.Perm (List.range (gridCount N blockSize)))
    (inputReadable : AccessMemory.Readable access inputBase (N * elementBytes encoding))
    (outputWritable : AccessMemory.Writable access outputBase (min N blockSize * 4))
    (distinct : x ≠ y) (separate : Separate encoding inputBase outputBase N)
    (indexDomain : IndexDomain N)
    (inputDomain : ByteDomain inputBase (elementBytes encoding) N)
    (outputDomain : ByteDomain outputBase 4 N) :
    ∃ final logicalFinal,
      AccessMemory.runSchedule M encoding inputBase outputBase N order access = .ok final ∧
      GatedGrid.runGrid (backend M cmp16 cmp32) (source encoding.profile x y N blockSize) N blockSize
        (stateFromBytes encoding access.bytes inputBase outputBase N x y initial) = some logicalFinal ∧
      LogicalView encoding final.bytes inputBase outputBase N x y logicalFinal.mem := by
  obtain ⟨final, logicalFinal, hs, _, he, _, _, _, hv, _⟩ :=
    relu_access_grid_correctness M cmp16 cmp32 encoding inputBase outputBase N x y access
      (stateFromBytes encoding access.bytes inputBase outputBase N x y initial) order complete inputReadable outputWritable
      (state_from_bytes_view encoding access.bytes inputBase outputBase N x y initial distinct)
      distinct separate indexDomain inputDomain outputDomain
  exact ⟨final, logicalFinal, hs, he, hv⟩

#axiomsClean relu_access_grid_correctness
#axiomsClean constructed_access_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.AccessGridCorrect
