import bench.tritonbench_g.matrix_reduction.Proofs.NativeHost
import bench.tritonbench_g.matrix_reduction.Proofs.TypedShape

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Native
open VeriTile.Triton FP.Structural TypedSource MachineMax ConcreteCorrect
noncomputable section
set_option maxHeartbeats 4000000

/-- Both original storage profiles retain the exact shared source statement
structure after dtype erasure. This is checked separately from native scalar
conformance and from reduction-tree tie choices. -/
theorem original_source_shape (d : Storage) (X Y : RegionName) (sm sn sy M N : Nat) :
    (TypedSource.kernel d X Y sm sn sy M N).eraseDType.surfaceBody =
    (load_reduce_kernel X Y sm sn sy M N).eraseDType.surfaceBody :=
  TypedSource.source_shape d X Y sm sn sy M N

/-- Physical native output words and the actual shared source interpreter's
stored output words have equal non-NaN value ranks. Native tree selection can
choose the other signed zero; the public maximum/Torch assertion still agrees.
Both executions are derived, rather than accepted as a successful-result oracle. -/
theorem native_source_correspondence (mem : Memory S d) (num : Numerics)
    (laws : NumericalContracts num) (admitted : Admitted M N)
    (p : Layout d M N) (valid : p.Valid) (schedule : Schedule M N)
    (initial : S) (xs : Fin M → Fin N → Bits d) (input : Input mem p initial xs)
    (hn : ∀ i j, isNaN d (xs i j) = Bool.false)
    (rest : Algebra Word) (source : State Word) (X Y : RegionName)
    (source_input : ∀ i j, source.mem X (i.val*max N 1+j.val) =
      .mk (channel d).toTileDType (ofFloat (channel d) (encode d (xs i j)))) :
    ∃ native sourceFinal,
      kernel mem num admitted p schedule initial = .ok native ∧
      FP.Structural.exec (backend rest) (TypedSource.kernel d X Y (max N 1) 1 1 M N) source = some sourceFinal ∧
      KernelSpec mem p initial native xs ∧
      (∀ i, ∃ vn vs,
        mem.load native (p.yAddress i) = some vn ∧
        sourceFinal.mem Y i.val = .mk (channel d).toTileDType (ofFloat (channel d) (encode d vs)) ∧
        IsMaximum d (xs i) vn ∧ IsMaximum d (xs i) vs ∧ rank d vn = rank d vs) := by
  obtain ⟨native,hk,hs⟩ := kernel_correct mem num laws admitted p valid schedule initial xs input hn
  obtain ⟨t,he,hv,_⟩ := ConcreteCorrect.source_correct d rest source X Y (max N 1) 1 1 M N
    admitted.cols xs (by simpa using source_input) hn
  refine ⟨native,t,hk,he,hs,?_⟩
  intro i
  obtain ⟨vn,hnative,hmaxn,_,_⟩ := hs.1 i
  obtain ⟨vs,hsource,hmaxs,_⟩ := hv i
  exact ⟨vn,vs,hnative,hsource,hmaxn,hmaxs,Host.maximum_rank_eq d (xs i) vn vs hmaxn hmaxs⟩

theorem signed_zero_rank_fp16 :
    rank .fp16 (0x0000 : BitVec 16) = rank .fp16 (0x8000 : BitVec 16) := by decide +kernel

theorem signed_zero_rank_fp32 :
    rank .fp32 (0x00000000 : BitVec 32) = rank .fp32 (0x80000000 : BitVec 32) := by decide +kernel

#axiomsClean original_source_shape
#axiomsClean native_source_correspondence
#axiomsClean signed_zero_rank_fp16
#axiomsClean signed_zero_rank_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Native
