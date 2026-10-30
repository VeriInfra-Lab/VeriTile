import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelReverseLoop
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardMiddle
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRecomputeLoop
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 20000000
variable {α : Type} [Inhabited α]

def start (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α) :=
  let reverse := KernelReverseMemory.memory M s f scale bh iv u s.T
  if f.initialGrad then KernelBackwardStateMemory.store M s "dh0" bh iv
    (KernelReverseMemory.carry M s f scale bh iv u s.T) reverse else reverse
def result (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α) :=
  KernelRecomputeReferenceMemory.memory M s f scale bh iv (start M s f scale bh iv u) s.T

theorem executes (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,FP.Structural.exec (engine M) (source .backward s f scale) u=some v ∧
      v.mem=(result M s f scale bh iv u).mem := by
  let prelude := KernelPrefix.backward M s f scale u
  obtain ⟨reverse,reverseSuccess,revInv⟩ := KernelReverseLoop.executes M s f scale bh iv u 0 prelude.val
    (KernelReverseLoop.initial M s f scale bh iv u hv hk hh)
  let middle := KernelBackwardMiddle.execute M s f scale bh iv
    (KernelReverseMemory.carry M s f scale bh iv u s.T) reverse revInv.carry revInv.static
  have middleMemory : middle.val.mem=(start M s f scale bh iv u).mem := by
    change (KernelBackwardMiddle.execute M s f scale bh iv _ reverse revInv.carry revInv.static).val.mem=_
    rw [KernelBackwardMiddle.memory]
    unfold start
    cases hf : f.initialGrad <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    · exact revInv.memory
    · exact KernelBackwardStateMemory.mem_congr M s "dh0" bh iv _ _ _ revInv.memory
  have seed : (KernelBackwardReference.input M s f bh iv reverse).initial=
      (KernelBackwardReference.input M s f bh iv (start M s f scale bh iv u)).initial := by
    simp only [KernelBackwardReference.input,KernelReference.read,revInv.memory]
    unfold start
    cases hf : f.initialGrad <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    simp only [KernelBackwardStateMemory.frame M s "dh0" bh iv _ _ "h0" _ (Or.inl (by decide))]
  have middleRegs := KernelBackwardMiddle.registers M s f scale bh iv
    (KernelReverseMemory.carry M s f scale bh iv u s.T) reverse revInv.carry revInv.static
  rw [seed] at middleRegs
  obtain ⟨v,recomputeSuccess,recInv⟩ := KernelRecomputeLoop.executes M s f scale bh iv
    (start M s f scale bh iv u) 0 middle.val ⟨Nat.zero_le _,middleRegs,middleMemory⟩
  have reverseRun : run (engine M) [KernelProgram.loop .backward s f scale] prelude.val=some reverse := by
    simp [KernelProgram.loop,run,step,evalComputeOp,evalOp_unfold,evalExpr,reverseSuccess]
  have recomputeRun : run (engine M) [KernelProgram.recompute s f scale] middle.val=some v := by
    simp only [KernelProgram.recompute,run,step,recomputeSuccess]
    rfl
  have finalNil : (second s f scale).after=[] := by
    cases hb : f.vectorBeta <;>
      simp only [second,first,source,Source.backward,hb,Bool.false_eq_true,ite_false,ite_true,
        ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
        String.reduceEq,beq_self_eq_true,↓reduceIte] <;> rfl
  refine ⟨v,?_,recInv.memory⟩
  change run (engine M) (source .backward s f scale).surfaceBody u=some v
  rw [source_body,List.append_assoc,run_append,prelude.property]
  simp only [Option.bind_some]
  rw [run_append,reverseRun]
  simp only [Option.bind_some]
  rw [backward_tail,List.append_assoc,run_append,middle.property]
  simp only [Option.bind_some]
  rw [run_append,recomputeRun]
  simp [finalNil,run]

#axiomsClean executes
#specNonCircular result avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackward
