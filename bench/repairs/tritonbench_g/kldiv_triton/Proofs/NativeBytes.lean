import bench.repairs.tritonbench_g.kldiv_triton.Proofs.NativeBridge

namespace VeriTile.Bench.Repairs.KldivTriton.NativeHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
open Native (Word)
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option autoImplicit false

def ForwardBytes {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (p : Native.Instructions) (host : TorchPorts)
    (call : ForwardCall) (initial : World M) (output : Matrix) (out : View) (after : World M) : Prop :=
  if call.mode=.none then ∀ i,i<output.view.numel → ∀ byte,byte<4 →
    ops.observe after.memory (out.key i).owner ((out.key i).address+byte)=
      (SwigluTriton.Words.signNegation.encode out.negative (forward ops p call initial i)).extractLsb' (8*byte) 8
  else ∃ plan : ReductionPlan output.view.numel, ∀ byte,byte<4 →
    ops.observe after.memory (out.key 0).owner ((out.key 0).address+byte)=
      (SwigluTriton.Words.signNegation.encode out.negative
        (scalarNormalize (arithmetic host) (denominator call.mode call.prediction.rows call.prediction.cols)
          (HostReduction.eval host.add (Native.word 0) (fun i => forward ops p call initial i.val) plan.tree))).extractLsb' (8*byte) 8

theorem forward_bytes {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Native.Instructions) (host : TorchPorts) (call : ForwardCall) (initial : World M)
    (output : Matrix) (out : View) (after : World M)
    (result : ForwardReturned ops SwigluTriton.Words.signNegation (arithmetic host) call output (forward ops p call initial) out after) :
    ForwardBytes ops p host call initial output out after := by
  have value := result.value
  unfold ForwardBytes
  by_cases mode : call.mode=.none
  · simp only [mode,if_true] at value ⊢
    intro i hi byte hb
    exact SwigluTriton.Words.logical_bytes ops codec after.memory out i _ (value.2 i hi) byte hb
  · simp only [mode,if_false] at value ⊢
    obtain ⟨plan,readback⟩ := value.value
    refine ⟨plan,?_⟩
    intro byte hb
    exact SwigluTriton.Words.logical_bytes ops codec after.memory out 0 _ readback byte hb

def BackwardBytes {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (p : Native.Instructions) (host : TorchPorts)
    (call : BackwardCall) (initial : World M) (out : View) (after : World M) : Prop :=
  (∀ i,i<call.input.view.numel → ∀ byte,byte<4 →
    ops.observe after.memory (call.input.view.key i).owner ((call.input.view.key i).address+byte)=
      (SwigluTriton.Words.signNegation.encode call.input.view.negative (backward ops p call initial i)).extractLsb' (8*byte) 8) ∧
  (if scalarOneCondition (arithmetic host) call.gradient.shape (logical ops SwigluTriton.Words.signNegation initial.memory call.gradient) then
    out=call.input.view else ∀ i,i<out.numel → ∀ byte,byte<4 →
      ops.observe after.memory (out.key i).owner ((out.key i).address+byte)=
        (SwigluTriton.Words.signNegation.encode out.negative (host.mul
          (backward ops p call initial (broadcastIndex call.input.view.shape out.shape i))
          (logical ops SwigluTriton.Words.signNegation initial.memory call.gradient (broadcastIndex call.gradient.shape out.shape i)))).extractLsb' (8*byte) 8)

theorem backward_bytes {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Native.Instructions) (host : TorchPorts) (call : BackwardCall) (initial : World M)
    (out : View) (after : World M)
    (result : BackwardReturned ops SwigluTriton.Words.signNegation (arithmetic host) call.input call.gradient.shape
      (backward ops p call initial) (logical ops SwigluTriton.Words.signNegation initial.memory call.gradient) out after) :
    BackwardBytes ops p host call initial out after := by
  constructor
  · intro i hi byte hb
    exact SwigluTriton.Words.logical_bytes ops codec after.memory call.input.view i _ (result.inputValue i hi) byte hb
  · have branch := result.branch
    by_cases scalar : scalarOneCondition (arithmetic host) call.gradient.shape (logical ops SwigluTriton.Words.signNegation initial.memory call.gradient)
    · simpa only [scalar,if_true] using branch
    · simp only [scalar,if_false] at branch ⊢
      intro i hi byte hb
      exact SwigluTriton.Words.logical_bytes ops codec after.memory out i _ (branch.2 i hi) byte hb

#axiomsClean forward_bytes
#axiomsClean backward_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.NativeHost
