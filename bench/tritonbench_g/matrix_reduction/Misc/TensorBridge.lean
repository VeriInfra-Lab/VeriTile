import bench.tritonbench_g.matrix_reduction.Proofs.KernelReal
import bench.tritonbench_g.matrix_reduction.Misc.NativeCorrect
import bench.tritonbench_g.matrix_reduction.Proofs.KernelNativeResults
import bench.tritonbench_g.matrix_reduction.Proofs.KernelAddress

/-! Actual input-Tensor storage applications of the complete raw kernel
results. Tensor observations and the optional non-NaN interpretation belong
here; they do not restrict either primary raw-kernel theorem. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.TensorBridge
open VeriTile.Triton FP.Structural TypedSource MachineMax ConcreteCorrect
noncomputable section

/-- The actual typed kernel computes an attained row maximum and preserves
 every exterior cell. Inputs are values at the generated Tensor's actual
 storage addresses. No host execution or random-sample condition is required.
 Positive N is necessary for tl.max; successful compilation is an API boundary. -/
theorem matrix_reduction_correct (storage : Storage) (M N : Nat) (hN : 0 < N)
    (x : InputStorage storage M N) (read : Nat → ℝ) (s : BlockState)
    (input : InputValues x read s) :
    VeriTile.Spec.Real (∃ t,
      exec (kernel storage "x" "out" N 1 1 M N).eraseDType.toAlgKernel s = some t ∧
      (∀ i, KernelMath.IsMaximum (fun j => read (x.address i j)) (t.readMem "out" i.val)) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o)) :=
  KernelReal.coordinates storage M N hN s (fun i j => read (x.address i j)) input

/-- Typed cells are the words at the generated input Tensor's actual addresses.
 No restriction on payloads is part of this storage binding. -/
def InputCells (x : InputStorage storage M N) (read : Nat → Bits storage)
    (s : State Word) : Prop :=
  ∀ i : Fin M, ∀ j : Fin N, s.mem "x" (i.val*N+j.val) =
    .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage (read (x.address i j))))

/-- Ordinary typed buffer cells for any floating primitive interpretation. -/
def SourceCells {α : Type} (x : InputStorage storage M N) (read : Nat → α)
    (s : State α) : Prop :=
  ∀ i : Fin M, ∀ j : Fin N, s.mem "x" (i.val*N+j.val) =
    .mk (channel storage).toTileDType (ofFloat (channel storage) (read (x.address i j)))

/-- There is no optimized source variant to invent. This is the successful
 original source / primitive-graph relation: every row output, its actual
 Tensor-storage inputs, and the full memory frame. Every floating primitive
 remains arbitrary, so NaNs, infinities and signed zeros require no guard. -/
theorem matrix_reduction_fp {α : Type} [Inhabited α] (A : Algebra α)
    (storage : Storage) (M N : Nat) (hN : 0 < N)
    (x : InputStorage storage M N) (read : Nat → α) (s : State α)
    (input : SourceCells x read s) :
    ∃ t, exec A (kernel storage "x" "out" N 1 1 M N) s = some t ∧
      (∀ i : Fin M, t.mem "out" i.val = .mk (channel storage).toTileDType
        (ofFloat (channel storage) (KernelFP.formula A storage M N
          (fun row col => read (x.address row col)) i))) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) :=
  KernelNativeResults.matrix_reduction_kernel_fp A storage M N hN s
    (fun row col => read (x.address row col)) input

/-- Additional non-NaN interpretation of the proved bit result. This theorem
 does not add a Python guard: arbitrary words remain executable above. Exact
 half promotion and inverse narrowing are proved bit lemmas, not experiments. -/
theorem matrix_reduction_maximum (storage : Storage) (rest : Algebra Word)
    (M N : Nat) (hN : 0 < N) (x : InputStorage storage M N)
    (read : Nat → Bits storage) (s : State Word) (input : InputCells x read s)
    (nonNaN : ∀ i j, isNaN storage (read (x.address i j)) = Bool.false) :
    ∃ t, exec (backend rest) (kernel storage "x" "out" N 1 1 M N) s = some t ∧
      (∀ i : Fin M, ∃ v : Bits storage,
        t.mem "out" i.val = .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage v)) ∧
        IsMaximum storage (fun j => read (x.address i j)) v ∧ isNaN storage v = Bool.false) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  apply source_correct storage rest s "x" "out" N 1 1 M N hN
    (fun i j => read (x.address i j))
  · intro i j
    simpa only [Nat.mul_one] using input i j
  · exact nonNaN

#axiomsClean matrix_reduction_correct
#axiomsClean matrix_reduction_fp
#axiomsClean matrix_reduction_maximum
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.TensorBridge
