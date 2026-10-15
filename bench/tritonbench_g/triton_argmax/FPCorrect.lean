import bench.tritonbench_g.triton_argmax.Proofs.BitCorrect
import bench.tritonbench_g.triton_argmax.Proofs.DimCorrect
import bench.tritonbench_g.triton_argmax.Proofs.TypedShape
namespace VeriTile.Bench.TritonBenchG.TritonArgmax.FPCorrect
open VeriTile.Triton FP.Structural TypedSource MatrixReduction.MachineMax BitCorrect
noncomputable section

specification argmax_stage1_bit_correctness
    (storage : Storage) (rest : Algebra (Bits storage))
    (s : State (Bits storage)) (X Y : RegionName) (I : Region .int)
    (N B : Nat) (index64 : Bool) (hB : 0 < B) (hYI : Y ≠ Region.cast I)
    (hnotNaN : ∀ i, isNaN storage (stage1Input storage s X N B i) = Bool.false) :
    ∃ t, FP.Structural.exec (BitBackend.backend storage rest)
        (stage1 storage X Y I N B index64) s = some t ∧
      (∃ j : Fin B, IsArgMaximum storage (stage1Input storage s X N B) j ∧
        t.mem Y (s.pids 0) = .mk (channel storage).toTileDType
          (ofFloat (channel storage) (stage1Input storage s X N B j)) ∧
        t.mem (Region.cast I) (s.pids 0) = .mk .int (Int.ofNat (j.val + s.pids 0 * B)) ∧
        isNaN storage (stage1Input storage s X N B j) = Bool.false) ∧
      (∀ r o, (r ≠ Y ∨ o ≠ s.pids 0) → (r ≠ Region.cast I ∨ o ≠ s.pids 0) →
        t.mem r o = s.mem r o) :=
  BitCorrect.argmax_stage1_bit_correctness storage rest s X Y I N B index64 hB hYI hnotNaN

#axiomsClean argmax_stage1_bit_correctness

specification argmax_stage2_bit_correctness
    (storage : Storage) (rest : Algebra (Bits storage))
    (s : State (Bits storage)) (X : RegionName) (I O : Region .int)
    (N B : Nat) (hB : 0 < B)
    (hnotNaN : ∀ i, isNaN storage (stage2Input storage s X N B i) = Bool.false) :
    ∃ t, FP.Structural.exec (BitBackend.backend storage rest) (stage2 storage X I O N B) s = some t ∧
      (∃ j : Fin B, IsArgMaximum storage (stage2Input storage s X N B) j ∧
        t.mem (Region.cast O) 0 = .mk .int ((s.mem (Region.cast I) j.val).read .int) ∧
        isNaN storage (stage2Input storage s X N B j) = Bool.false) ∧
      (∀ r o, r ≠ Region.cast O ∨ o ≠ 0 → t.mem r o = s.mem r o) :=
  BitCorrect.argmax_stage2_bit_correctness storage rest s X I O N B hB hnotNaN

#axiomsClean argmax_stage2_bit_correctness

/-- All rows of the original dim-specific source in its accepted nonempty
single-block regime; both storage profiles and index-width branches. -/
specification argmax_dim_single_block_bit_correctness
    (storage : Storage) (rest : Algebra MatrixReduction.ConcreteCorrect.Word)
    (s : State MatrixReduction.ConcreteCorrect.Word)
    (X : RegionName) (O : Region .int) (M N K BM BN : Nat) (index64 : Bool)
    (hN : 0 < N) (hBN : N ≤ BN)
    (hOutInj : Function.Injective (DimCorrect.outputOffset s K BM))
    (hnotNaN : ∀ i j, isNaN .fp32 (DimCorrect.input storage s X M N K BM BN i j) = Bool.false) :
    ∃ t, FP.Structural.exec (DimBackend.backend rest) (dim storage X O M N K BM BN index64) s = some t ∧
      (∀ i : Fin BM, DimCorrect.row s BM i < M → ∃ j : Fin BN,
        j.val < N ∧ IsArgMaximum .fp32 (DimCorrect.input storage s X M N K BM BN i) j ∧
        t.mem (Region.cast O) (DimCorrect.outputOffset s K BM i) = .mk .int (Int.ofNat j.val)) ∧
      (∀ r o, (r ≠ Region.cast O ∨ ∀ i : Fin BM,
        DimCorrect.row s BM i < M → o ≠ DimCorrect.outputOffset s K BM i) →
        t.mem r o = s.mem r o) :=
  DimCorrect.source_correct storage rest s X O M N K BM BN index64 hN hBN hOutInj hnotNaN

#axiomsClean argmax_dim_single_block_bit_correctness

end
end VeriTile.Bench.TritonBenchG.TritonArgmax.FPCorrect
