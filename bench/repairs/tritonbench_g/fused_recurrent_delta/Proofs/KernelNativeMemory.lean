import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelNativeBackwardSpec
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardFacts
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeMemory
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters KernelNativeCarrier
namespace N
export KernelNativeBackwardSpec (input enhance reverse forward dk dv dbeta dq betaSum correction)
end N
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000

def enhanced (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (n : Nat) :=
  let a := N.input isa routing s f bh iv u
  N.enhance s scale a (s.T-1-n) (N.reverse routing s scale a n)
def reverseMemory (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) : Nat → State Node
  | 0 => u
  | n+1 =>
    let M := algebra isa routing scale
    let a := N.input isa routing s f bh iv u
    let t := s.T-1-n
    let g := enhanced isa routing scale s f bh iv u n
    KernelBackwardMemory.betaStore M s f bh iv t (N.dbeta routing s a t g)
      (KernelMemory.rowStore M s "dv" bh iv s.BBV t (N.dv routing s a t g)
        (KernelBackwardMemory.keyStore M s "dk" (plane s bh iv) t (N.dk routing s a t g)
          (reverseMemory isa routing scale s f bh iv u n)))
def start (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) :=
  let previous := reverseMemory isa routing scale s f bh iv u s.T
  if f.initialGrad then KernelBackwardStateMemory.store (algebra isa routing scale) s "dh0" bh iv
    (N.reverse routing s scale (N.input isa routing s f bh iv u) s.T) previous else previous
def query (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (t : Nat) :=
  let a := N.input isa routing s f bh iv (start isa routing scale s f bh iv u)
  N.dq routing s scale a t (N.forward s a (t+1))
def corrected (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (t : Nat) :=
  let previous := start isa routing scale s f bh iv u
  N.correction routing s (N.forward s (N.input isa routing s f bh iv previous) t)
    (fun i => KernelNativeSpec.read isa routing previous "dk" (keyAddress s (plane s bh iv) t i.val) (decide (i.val<s.K)))
    (fun j => KernelNativeSpec.read isa routing previous "dv" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V)))

theorem enhanced_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (n : Nat) :
    KernelReverseMemory.enhanced (algebra isa routing scale) s f scaleSlot bh iv u n=
      enhanced isa routing scale s f bh iv u n := by
  unfold KernelReverseMemory.enhanced KernelReverseMemory.carry
  rw [KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.reverse_lowering,
    KernelNativeBackwardSpec.enhance_lowering]
  rfl
theorem reverse_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (n : Nat) :
    KernelReverseMemory.memory (algebra isa routing scale) s f scaleSlot bh iv u n=
      reverseMemory isa routing scale s f bh iv u n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold KernelReverseMemory.memory reverseMemory
    rw [ih,enhanced_lowering,KernelNativeBackwardSpec.input_lowering]
    rfl
theorem start_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) :
    KernelBackward.start (algebra isa routing scale) s f scaleSlot bh iv u=start isa routing scale s f bh iv u := by
  unfold KernelBackward.start start KernelReverseMemory.carry
  rw [reverse_lowering,KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.reverse_lowering]
theorem query_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (t : Nat) :
    KernelRecomputeReferenceMemory.query (algebra isa routing scale) s f scaleSlot bh iv
      (KernelBackward.start (algebra isa routing scale) s f scaleSlot bh iv u) t=
      query isa routing scale s f bh iv u t := by
  unfold KernelRecomputeReferenceMemory.query KernelRecomputeReferenceMemory.carry
  rw [start_lowering,KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.forward_lowering,
    KernelNativeBackwardSpec.dq_lowering]
  rfl
theorem corrected_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (t : Nat) :
    KernelRecomputeReferenceMemory.corrected (algebra isa routing scale) s f bh iv
      (KernelBackward.start (algebra isa routing scale) s f scaleSlot bh iv u) t=
      corrected isa routing scale s f bh iv u t := by
  unfold KernelRecomputeReferenceMemory.corrected KernelRecomputeReferenceMemory.carry
  rw [start_lowering,KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.forward_lowering]
  unfold KernelRecomputeValues.dk corrected
  funext i
  simp [KernelBackwardReference.correction,KernelReference.sub,KernelReference.mul,KernelReference.read,
    KernelReference.cast,KernelReference.zero,KernelBackwardReference.valueSum,N.correction,
    KernelNativeBackwardSpec.correction,KernelNativeBackwardSpec.valueSum,KernelNativeSpec.read,algebra,scaleSlot]

#axiomsClean enhanced_lowering
#axiomsClean reverse_lowering
#axiomsClean start_lowering
#axiomsClean query_lowering
#axiomsClean corrected_lowering
#specNonCircular start avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelBackward.start]
#specNonCircular corrected avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelRecomputeReferenceMemory.corrected]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeMemory
