import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.IOEquiv

/-! The actual addition-order variant under the source-selected fused native
profile. Either source consumes alpha/data/grad in the same FMA order. FP32
primitive semantics are the existing model boundary; no floating rewrite or
new numerical conformance experiment is assumed. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.NativeCorrect
open VeriTile.Triton FP.Structural
noncomputable section

/-- IO.lean declares the grad/data/signed-mask channels, windows, vetoed store
mask and actual FP32 source invocation with its runtime scalar ABI word. -/
def original (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) : ProgramIO Graph.Expr Graph.Expr :=
  NativeIO.originalIO rest alphaWord storage n pid token

/-- Existing product-left source; it shares the original IO and native profile. -/
def commuted (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) :
    ProgramIO.Variant (original rest alphaWord storage n pid token) :=
  NativeIO.candidate rest alphaWord storage n pid token

/-- Both actual sources safely execute and agree on every active output cell,
with exact exterior frames. Only active grad operands must be input word leaves;
data expressions, vetoed grad values and every FP32 word remain admitted. -/
theorem masked_add_commuted_equiv (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) :
    (original rest alphaWord storage n pid token).EquivalentOn
      (commuted rest alphaWord storage n pid token) (NativeIO.LeafGradient n pid) :=
  NativeIO.equivalent rest alphaWord storage n pid token

#axiomsClean masked_add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.NativeCorrect

/-! Optional single-source native-word application, retaining its original name. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.KernelNativeResults
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
noncomputable section

/-- Independent word result under the fixed fused primitive, including vetoed
NaN payloads and signed zeros. The primitive itself remains a modeling boundary. -/
def reference (fma : Word → Word → Word → Word) (alpha grad data : Word) (mask : Int) : Word :=
  if mask = 0 then fma alpha data grad else grad

theorem masked_add_kernel_result (rest : Algebra Graph.Expr) (ops : Graph.Operations)
    (storage : MaskStorage) (n : Nat) (token : ℝ) (alphaWord : Word) (before : State Graph.Expr)
    (grad data : Nat → Word) (mask : Nat → Int)
    (hg : ∀ i : TileIndex [blockSize], offset (parameters "grad" "data" "mask" n) before i < n →
      (before.mem "grad" (offset (parameters "grad" "data" "mask" n) before i)).read .real =
        .leaf (grad (offset (parameters "grad" "data" "mask" n) before i)))
    (hp : ∀ i : TileIndex [blockSize], offset (parameters "grad" "data" "mask" n) before i < n →
      (before.mem "data" (offset (parameters "grad" "data" "mask" n) before i)).read .real =
        .leaf (data (offset (parameters "grad" "data" "mask" n) before i)))
    (hm : ∀ i : TileIndex [blockSize], offset (parameters "grad" "data" "mask" n) before i < n →
      (before.mem "mask" (offset (parameters "grad" "data" "mask" n) before i)).read .int =
        mask (offset (parameters "grad" "data" "mask" n) before i)) :
    ∃ final, FP.Structural.exec (Graph.nativeAlgebra rest alphaWord)
        (kernel storage "grad" "data" "mask" n token) before = some final ∧
      (∀ i : TileIndex [blockSize], offset (parameters "grad" "data" "mask" n) before i < n →
        Graph.eval ops ((final.mem "grad" (offset (parameters "grad" "data" "mask" n) before i)).read .real) =
          reference ops.fmaRN32 alphaWord (grad (offset (parameters "grad" "data" "mask" n) before i))
            (data (offset (parameters "grad" "data" "mask" n) before i))
            (mask (offset (parameters "grad" "data" "mask" n) before i))) ∧
      (∀ (r : RegionName) address, (r ≠ "grad" ∨ ∀ i : TileIndex [blockSize],
        active (parameters "grad" "data" "mask" n) before i →
          address ≠ offset (parameters "grad" "data" "mask" n) before i) →
        final.mem r address = before.mem r address) := by
  let p := parameters "grad" "data" "mask" n
  let M := Graph.nativeAlgebra rest alphaWord
  obtain ⟨final, he, memory⟩ := complete_correct M storage p token before
  refine ⟨final, he, ?_, ?_⟩
  · intro i hi
    have hg' : (before.mem p.output (offset p before i)).read .real =
        .leaf (grad (offset p before i)) := hg i hi
    have hp' : (before.mem p.x (offset p before i)).read .real =
        .leaf (data (offset p before i)) := hp i hi
    have hm' : (before.mem p.y (offset p before i)).read .int = mask (offset p before i) := hm i hi
    change Graph.eval ops ((final.mem p.output (offset p before i)).read .real) =
      reference ops.fmaRN32 alphaWord (grad (offset p before i))
        (data (offset p before i)) (mask (offset p before i))
    rw [memory]
    by_cases zero : mask (offset p before i) = 0
    · have ha : active p before i := ⟨hi, hm'.trans zero⟩
      rw [expected_readback M p token before i ha]
      simp [resultCell, updateValue, alphaValue, M, Graph.nativeAlgebra, Graph.algebra,
        Graph.binary, Graph.eval, hg', hp', reference, zero]
    · have outside : ∀ j, active p before j → offset p before i ≠ offset p before j := by
        intro j hj same
        have hz := hj.2
        rw [←same, hm'] at hz
        exact zero hz
      rw [expected_frame M p token before p.output (offset p before i) (Or.inr outside)]
      simp [hg', Graph.eval, reference, zero]
  · intro r address outside
    rw [memory]
    exact expected_frame M p token before r address outside

#specNonCircular reference avoiding [kernel, commutedKernel, FP.Structural.exec]
#axiomsClean masked_add_kernel_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.KernelNativeResults
