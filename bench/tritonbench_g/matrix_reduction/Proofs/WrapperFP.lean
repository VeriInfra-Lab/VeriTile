import bench.tritonbench_g.matrix_reduction.Proofs.ConcreteCorrect
import bench.tritonbench_g.matrix_reduction.Proofs.TypedShape
import bench.tritonbench_g.matrix_reduction.Proofs.Host
import bench.tritonbench_g.matrix_reduction.Proofs.NativeCorrespondence

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

/-- Complete production-wrapper composition for both source-declared storage
formats, including all fallible setup stages and the empty-axis failure.
The substantive FP statement is bit/max refinement of the original program;
there is no invented optimized candidate. The native entrypoint imported above
adds physical-address and arbitrary-tree refinement under explicit contracts. -/
specification load_reduce_source_outcomes (rest : Algebra Word)
    (M N : Nat) (dtype : String) (ops : (storage : Storage) → Host.Ops storage M N)
    (alloc : ∀ storage, Host.AllocationContracts storage M N (ops storage))
    (torch : ∀ storage, Host.TorchContracts storage M N (ops storage))
    (samples : ∀ storage, Host.SamplesNonNaN storage M N (ops storage)) :
    Host.FunctionSpec rest M N dtype ops (Host.loadReduce rest M N dtype ops) :=
  Host.load_reduce_outcomes rest M N dtype ops alloc torch samples

#axiomsClean matrix_reduction_machine_correctness
#axiomsClean load_reduce_source_outcomes
#axiomsClean Native.native_source_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Native
open VeriTile.Triton TypedSource MachineMax ConcreteCorrect
open Host (selectDType)
noncomputable section

/-- Whole original dtype dispatcher and implicit None return, with all source
stages and actual native byte addresses. Numerical assumptions and ordinary
runtime/Torch contracts remain explicit arguments; no numerical validation or
compiler correctness proof is silently inferred from this theorem. -/
specification load_reduce_native_outcomes (mem : (d : Storage) → Memory S d) (num : Numerics)
    (laws : NumericalContracts num) (M N : Nat) (dtype : String) (ops : (d : Storage) → Ops S d M N)
    (alloc : ∀ d, AllocationContracts (mem d) M N (ops d))
    (torch : ∀ d, TorchContracts (mem d) M N (ops d))
    (samples : ∀ d, SamplesNonNaN (mem d) M N (ops d)) :
    FunctionSpec mem num M N dtype ops (loadReduce mem num M N dtype ops) := by
  cases hd : selectDType dtype with
  | error e => simp [FunctionSpec,hd,loadReduce,Except.bind,bind,pure]
  | ok d =>
    simp only [FunctionSpec,hd]
    constructor
    · exact full_native_outcomes (mem d) num laws M N (ops d) (alloc d) (torch d) (samples d)
    · simp [loadReduce,hd,Except.bind,Except.map,Except.pure,bind,pure,Functor.map]

#axiomsClean load_reduce_native_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Native
