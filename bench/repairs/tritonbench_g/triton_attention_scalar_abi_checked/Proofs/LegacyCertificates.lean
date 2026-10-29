import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Spec
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Contract

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.LegacyCertificates
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-! Exact prior operational statements, retained as ordinary helper theorems.
The public reader entries use the finite shared IO judgment. Both are tied to
the same complete source execution by PublicIO.real_schedule/native_schedule. -/

/-- Complete execution of the three shared readable kernel quotations.
The wrapper supplies canonical storage, input values, fresh disjoint scratch,
zero-initialized DQ, and the three full launch schedules. All saved values and
all output values are established by the execution proof. -/
theorem kernel_real (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Erased.R) (μ : RegionName → Nat → ℝ)
    (inputs : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,(mem r o).read .fp32=some (μ r o))
    (zero : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (some 0)) :
    Spec.Real (∃ final,Grid.wholeRun Erased.algebra Erased.model Erased.model Erased.model c scale fw pw bw mem=some final ∧
      Erased.TensorValues c scale μ final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o)) :=
  Erased.whole_execution_tensors c scale fw pw bw hf hp hb mem μ inputs zero

/-- Bound v2 kernel certificate. The complete v1 structural execution proof
is reused only after the raw scalar, source scale leaf and specialization are
joined by the local scalar-input law. -/
theorem kernel_native (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ → Native.Word) (binding : Native.ScalarBinding scalar literal)
    (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Native.Word) (μ : RegionName → Nat → Native.Word)
    (hi : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,mem r o=.mk .fp32 (μ r o))
    (hz : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 0) :
    PublicIO.SourceScale scalar I literal ∧
    ∃ final,Native.Selected.nativeRun (Native.constexprScale scalar) c (scalar.project scalar.raw)
        owners I literal fw pw bw mem=some final ∧
      Native.Selected.WordValues (Native.constexprScale scalar) c (scalar.project scalar.raw) owners I literal μ final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) :=
  ⟨fun kind owner=>binding.source_scale_word scalar literal I kind owner,
    Native.Selected.whole_execution_words (Native.constexprScale scalar) c (scalar.project scalar.raw)
      owners I literal fw pw bw hf hp hb mem μ hi hz⟩

#axiomsClean kernel_real
#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.LegacyCertificates
