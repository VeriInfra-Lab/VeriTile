import bench.tritonbench_g.matrix_reduction.Misc.KernelAllWords

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelNativeResults
open VeriTile.Triton FP.Structural TypedSource MachineMax ConcreteCorrect
noncomputable section

/-- Complete typed-source execution and independent ordered native graph on
raw input arrays, with all output cells and the entire exterior memory frame.
Floating primitive interpretations and input words remain unrestricted. -/
theorem matrix_reduction_kernel_fp {α : Type} [Inhabited α] (A : Algebra α)
    (storage : Storage) (M N : Nat) (hN : 0 < N) (s : State α)
    (xs : Fin M → Fin N → α)
    (input : ∀ i : Fin M, ∀ j : Fin N, s.mem "x" (i.val*N+j.val) =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (xs i j))) :
    ∃ t, exec A (kernel storage "x" "out" N 1 1 M N) s = some t ∧
      (∀ i : Fin M, t.mem "out" i.val = .mk (channel storage).toTileDType
        (ofFloat (channel storage) (KernelFP.formula A storage M N xs i))) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t,he,hv,hf⟩ := KernelFP.native_graph A storage s M N hN
  refine ⟨t,he,?_,hf⟩
  have values : KernelFP.sourceValues storage M N s = xs := by
    funext row col
    unfold KernelFP.sourceValues
    rw [input row col]
    cases storage <;> rfl
  intro i
  rw [hv,values]

/-- Supplementary explicit binary16/binary32 reference execution. Its chosen
 software NaN ordering is not asserted to be a native Triton NaN policy. -/
theorem matrix_reduction_bits (storage : Storage) (rest : Algebra Word)
    (M N : Nat) (hN : 0 < N) (s : State Word) :
    ∃ t, exec (backend rest) (kernel storage "x" "out" N 1 1 M N) s = some t ∧
      (∀ i : Fin M, t.mem "out" i.val = .mk (channel storage).toTileDType
        (ofFloat (channel storage) (output storage s "x" N 1 M N i))) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) :=
  KernelFP.all_words storage rest s M N hN

#axiomsClean matrix_reduction_kernel_fp
#axiomsClean matrix_reduction_bits
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelNativeResults
