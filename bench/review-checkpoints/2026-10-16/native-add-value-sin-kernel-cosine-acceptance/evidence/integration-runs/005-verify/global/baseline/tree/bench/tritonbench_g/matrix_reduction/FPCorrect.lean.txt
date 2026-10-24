import bench.tritonbench_g.matrix_reduction.Proofs.ConcreteCorrect
import bench.tritonbench_g.matrix_reduction.Proofs.TypedShape

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect
open VeriTile.Triton FP.Structural TypedSource MachineMax ConcreteCorrect
noncomputable section

/-- Full source execution in binary16/binary32 encodings with actual fp32
intermediate registers. Independent input-bit maximum, non-NaN and frame
properties; exact half widening/narrowing and order preservation are proved.
No hardware/compiler refinement is inferred from this source-side theorem. -/
specification matrix_reduction_machine_correctness
    (storage : Storage) (rest : Algebra Word)
    (s : State Word) (X Y : RegionName) (sm sn sy M N : Nat) (hN : 0 < N)
    (xs : Fin M → Fin N → Bits storage)
    (hinput : ∀ i j, s.mem X (i.val * sm + j.val * sn) =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage (xs i j))))
    (hnotNaN : ∀ i j, isNaN storage (xs i j) = Bool.false) :
    ∃ t, FP.Structural.exec (backend rest) (kernel storage X Y sm sn sy M N) s = some t ∧
      (∀ i : Fin M, ∃ v : Bits storage,
        t.mem Y i.val = .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage v)) ∧
        IsMaximum storage (xs i) v ∧ isNaN storage v = Bool.false) ∧
      (∀ r o, r ≠ Y ∨ M ≤ o → t.mem r o = s.mem r o) :=
  ConcreteCorrect.source_correct storage rest s X Y sm sn sy M N hN xs hinput hnotNaN

#axiomsClean matrix_reduction_machine_correctness
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect
