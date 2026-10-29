import bench.tritonbench_g.triton_conv2d_fwd.FPCorrect
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultSoftware

/-! Both original constexpr fp16 branches over fp32 input/weight/output storage.
The default false branch has a separately elaborated fp32 dot, with no operand
half cast. The original true branch and its historical contracts are preserved.
Each branch covers the complete triple loop and arbitrary masked-load undef;
no zero-padding, finite-input, single-tile-size or numerical-identity premise
is introduced. Native index-width, compiler fusion and dot-plan correspondence
remain external. Host grid composition and other storage dtypes are separate.
-/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.BranchesCorrect
open VeriTile.Triton FP.Structural
noncomputable section

/-- Original constexpr specialization, including the wrapper default false. -/
def source (useHalf : Bool) (p : FPValue.Parameters) : ComputeKernel :=
  if useHalf then FPValue.source p else FPDefault.source p

/-- Backend primitives are typed for the selected source operand format. -/
def backend {α : Type} (M : Algebra α) (halfDot : FPValue.Dot α)
    (fullDot : FPDefault.Dot α) (useHalf : Bool) : Algebra α :=
  if useHalf then FPValue.backend M halfDot else FPDefault.backend M fullDot

def expected {α : Type} [Inhabited α] (M : Algebra α) (halfDot : FPValue.Dot α)
    (fullDot : FPDefault.Dot α) (useHalf : Bool) (p : FPValue.Parameters)
    (initial : State α) : FPValue.CIndex p → α :=
  if useHalf then FPValue.outputValue M halfDot p initial
  else FPDefault.outputValue M fullDot p initial

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α) (halfDot : FPValue.Dot α)
    (fullDot : FPDefault.Dot α) (useHalf : Bool) (p : FPValue.Parameters)
    (initial : State α) : RegionName → Nat → Cell α :=
  if useHalf then (FPValue.written M halfDot p initial).mem
  else (FPDefault.written M fullDot p initial).mem

/-- All three source loops and both fp16/tf32 flag choices. Only the original
output-address injectivity condition is needed for pointwise readback. -/
specification all_fp16_branches_backend_correctness
    {α : Type} [Inhabited α] (M : Algebra α) (halfDot : FPValue.Dot α)
    (fullDot : FPDefault.Dot α) (useHalf : Bool) (p : FPValue.Parameters)
    (initial : State α)
    (hw : ∀ i j, FPValue.active p initial i → FPValue.active p initial j →
      FPValue.outputOffset p initial i = FPValue.outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M halfDot fullDot useHalf) (source useHalf p) initial = some final ∧
      final.mem = expectedMemory M halfDot fullDot useHalf p initial ∧
      FPValue.Post p initial final (expected M halfDot fullDot useHalf p initial) := by
  cases useHalf
  · exact FPDefault.correctness M fullDot p initial hw
  · exact FPValue.correctness M halfDot p initial hw

/-- Concrete software arithmetic for each branch: false keeps fp32 operands,
true rounds operands to binary16 before dot. Both use the explicitly defined
ascending dot order and outer fp32 carry additions. Every used callback is
closed, independently of rest; native TF32/fusion binding is not inferred. -/
specification all_fp16_branches_software_correctness
    (rest : Algebra FPValue.Word) (useHalf : Bool) (p : FPValue.Parameters)
    (initial : State FPValue.Word)
    (hw : ∀ i j, FPValue.active p initial i → FPValue.active p initial j →
      FPValue.outputOffset p initial i = FPValue.outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec
        (if useHalf then FPValue.backend (FPValue.software rest) FPValue.sequentialDot
         else FPDefault.backend (FPDefault.software rest) FPDefault.sequentialDot)
        (source useHalf p) initial = some final ∧
      FPValue.Post p initial final
        (if useHalf then FPValue.bitOutput p initial else FPDefault.bitOutput p initial) := by
  cases useHalf
  · exact FPDefault.software_correctness rest p initial hw
  · exact FPValue.software_correctness rest p initial hw

#specNonCircular expectedMemory avoiding [source, FP.Structural.exec]
#axiomsClean all_fp16_branches_backend_correctness
#axiomsClean all_fp16_branches_software_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.BranchesCorrect
