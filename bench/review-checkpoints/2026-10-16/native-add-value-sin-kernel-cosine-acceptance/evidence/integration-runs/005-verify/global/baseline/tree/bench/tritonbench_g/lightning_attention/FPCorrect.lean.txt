import bench.tritonbench_g.lightning_attention.Proofs.FPShape
import bench.tritonbench_g.lightning_attention.Proofs.FPWitness

/-! Full previously accepted semantic stages, each from independent initial
memory with successful typed execution, readback and frame. The three dot
contracts retain a complete backend parameter. Their separate sequential
software specializations are parameterized bit proofs, not native tensor-core
equivalences. The original outer forward/backward loops remain lowering-only. -/
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

specification kv_fp32_dot_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (kvSource p) s=some final ∧
      Post p.output (FPValue.kvOffset p) (fun _ => Bool.true) (wordKV cfg D p s) s final :=
  kv_software_correctness rest cfg D p s
#axiomsClean kv_fp32_dot_values

specification kv_sequential_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) (softwareDot cfg)) (kvSource p) s=some final ∧
      Post p.output (FPValue.kvOffset p) (fun _ => Bool.true) (wordKV cfg (softwareDot cfg) p s) s final :=
  kv_software_correctness rest cfg (softwareDot cfg) p s
#axiomsClean kv_sequential_bits

specification inter_fp32_dot_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (interSource p) s=some final ∧
      Post p.output (interOffset p) (fun _ => Bool.true) (wordInter D p s) s final :=
  inter_software_correctness rest cfg D p s
#axiomsClean inter_fp32_dot_values

specification inter_sequential_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) (softwareDot cfg)) (interSource p) s=some final ∧
      Post p.output (interOffset p) (fun _ => Bool.true) (wordInter (softwareDot cfg) p s) s final :=
  inter_software_correctness rest cfg (softwareDot cfg) p s
#axiomsClean inter_sequential_bits

specification body_fp32_dot_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) (hwidth : p.columns ≤ p.valueDim) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (bodySource p) s=some final ∧
      Post p.output (bodyOffset p s) (bodyActive p) (wordBody cfg D p s) s final :=
  body_software_correctness rest cfg D p s hwidth
#axiomsClean body_fp32_dot_values

specification body_sequential_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) (hwidth : p.columns ≤ p.valueDim) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) (softwareDot cfg)) (bodySource p) s=some final ∧
      Post p.output (bodyOffset p s) (bodyActive p) (wordBody cfg (softwareDot cfg) p s) s final :=
  body_software_correctness rest cfg (softwareDot cfg) p s hwidth
#axiomsClean body_sequential_bits

specification gradient_payload_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) (hinj : Function.Injective (gradOffset p s)) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) (softwareDot cfg)) (gradSource p) s=some final ∧
      Post p.output (gradOffset p s) (gradActive p s) (wordGrad p s) s final :=
  grad_software_correctness rest cfg (softwareDot cfg) p s hinj
#axiomsClean gradient_payload_bits

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
theorem kv_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (kvSource witnessParameters) witnessInitial=some final ∧
      final.mem "O" 0=.mk .fp32 (FP.Value.ofBits .fp32 0x41a00000) ∧
      final.mem "O" 1=.mk .fp32 (FP.Value.ofBits .fp32 0x42040000) ∧
      final.mem "Q" 4=witnessInitial.mem "Q" 4 := FPValue.kv_bit_success rest
#axiomsClean kv_bit_success

theorem inter_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (interSource witnessParameters) witnessInitial=some final ∧
      final.mem "O" 0=.mk .fp32 (FP.Value.ofBits .fp32 0x42200000) ∧
      final.mem "O" 1=.mk .fp32 (FP.Value.ofBits .fp32 0x42700000) ∧
      final.mem "Q" 4=witnessInitial.mem "Q" 4 := FPValue.inter_bit_success rest
#axiomsClean inter_bit_success

theorem body_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (bodySource witnessParameters) witnessInitial=some final ∧
      final.mem "O" 10=.mk .fp32 (FP.Value.ofBits .fp32 0x42280000) ∧
      final.mem "O" 11=.mk .fp32 (FP.Value.ofBits .fp32 0x427c0000) ∧
      final.mem "Q" 4=witnessInitial.mem "Q" 4 := FPValue.body_bit_success rest
#axiomsClean body_bit_success

theorem body_tail_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (bodySource tailParameters) tailInitial=some final ∧
      final.mem "O" 6=.mk .fp32 (FP.Value.ofBits .fp32 0x40000000) ∧
      final.mem "O" 10=tailInitial.mem "O" 10 := FPValue.body_tail_bit_success rest
#axiomsClean body_tail_bit_success

theorem grad_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (gradSource gradParameters) witnessInitial=some final ∧
      final.mem "O" 20=.mk .fp32 (FP.Value.ofBits .fp32 0x80000000) ∧
      final.mem "O" 21=.mk .fp32 (FP.Value.ofBits .fp32 0x7fc01234) ∧
      final.mem "O" 24=witnessInitial.mem "O" 24 := FPValue.grad_bit_success rest
#axiomsClean grad_bit_success

end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPCorrect
