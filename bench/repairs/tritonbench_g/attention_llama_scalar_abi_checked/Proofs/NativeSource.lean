import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Execution
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Tensor

/-! Public native precision results for the complete shared source. Primitive
semantics are scalar FP32, FP16 conversion, E5M2 conversion, exp2 and fixed K16
MMA operations. Their actual hardware meanings and the recorded SM80 selection
are ordinary primitive/compiler boundaries. No local numerical agreement or
whole-graph/output correspondence is a theorem premise. The mathematical Real
result is separately exposed in Correct.lean. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquivalence
open VeriTile Triton
open Source
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule HostGrid HostMemory
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def supportedHead (D : Nat) : Prop := D=16 ∨ D=32 ∨ D=64 ∨ D=128

/-- Successful whole source, independent native row outputs, and every
complementary typed memory cell. Input bindings contain coordinates only. -/
theorem attention_native_source (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : FP.Structural.State (Term α))
    (_hD : supportedHead c.D) (hN : 0<c.N) (bytes : ByteRange s) (hinj : ActiveInjective c s)
    (input : Fin 64→Target.Inputs α c.D)
    (binding : ∀ r, rows s r<c.H → Native.Binding c s r (input r)) :
    ∃ f, FP.Structural.exec algebra (kernel c) (bindScale s (Native.argument C arg))=some f ∧
      (∀ r d, rows s r<c.H →
        Native.lower A c.fp8 c.D (lane d.val) ((f.mem c.Out (address c s (r,d,()))).read .fp16)=
          Target.output A C (input r) arg c.fp8 c.causal c.N c.SP (s.pids 0) r.val d) ∧
      (∀ r o, (r≠c.Out ∨ ∀ i : TileIndex [64,c.D], rows s i.1<c.H → o≠address c s i) →
        f.mem r o=s.mem r o) := by
  obtain ⟨f,he,hv,hf⟩ := values_and_frame c s (Native.argument C arg) bytes hinj
  refine ⟨f,he,?_,hf⟩
  intro r d hr
  rw [hv (r,d,()) hr,FP.Structural.Cell.read_same]
  exact Native.output_binding A C arg c s r (input r) (binding r hr) hr hN d

#axiomsClean attention_native_source
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquivalence
