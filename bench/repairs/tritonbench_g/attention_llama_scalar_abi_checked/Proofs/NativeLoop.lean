import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeCarry

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
open VeriTile Triton
open Source
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000

/-- Full native recurrence, derived by induction from the shared source
iteration. No successful-run or attention-result premise is used. -/
theorem run_binding (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (hr : rows s r<c.H) (n : Nat) :
    CarryBinding A c r (accumulated c s (argument C arg) n)
      (Target.run A C input arg c.fp8 c.causal c.N c.SP (rows s r) n) := by
  induction n with
  | zero => exact initial_binding A c r
  | succ n ih => exact step_binding A C arg c s r input h hr n _ ih

theorem output_cast (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (scale : Term α) (n : Nat) (hn : 0<n) (r : Fin 64) (d : Fin c.D) :
    lower A c.fp8 c.D (lane d.val) (.half ((accumulated c s scale n).accumulator (r,d,())))=
      A.half (lower A c.fp8 c.D (lane d.val) ((accumulated c s scale n).accumulator (r,d,()))) := by
  cases n with
  | zero => omega
  | succ n => rfl

theorem count_positive (c : Config) (pid : Nat) (hN : 0<c.N) :
    0<MaskCoverage.count (MaskCoverage.stop c.N c.causal c.SP pid) := by
  cases hc : c.causal <;> simp [MaskCoverage.stop,hc,MaskCoverage.count] <;> omega

theorem output_binding (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (hr : rows s r<c.H) (hN : 0<c.N) (d : Fin c.D) :
    lower A c.fp8 c.D (lane d.val) (.half ((accumulated c s (argument C arg)
      (MaskCoverage.count (MaskCoverage.stop c.N c.causal c.SP (s.pids 0)))).accumulator (r,d,())))=
      Target.output A C input arg c.fp8 c.causal c.N c.SP (s.pids 0) r.val d := by
  rw [output_cast A c s _ _ (count_positive c _ hN)]
  exact congrArg A.half ((run_binding A C arg c s r input h hr _).accumulator d)

#axiomsClean run_binding
#axiomsClean output_binding
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
