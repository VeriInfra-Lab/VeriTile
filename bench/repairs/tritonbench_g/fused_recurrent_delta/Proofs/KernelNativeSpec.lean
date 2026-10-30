import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelNativeCarrier
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeSpec
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters KernelNativeCarrier
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000

def read (isa : Instructions) (routing : Routing) (u : State Node) (r : RegionName) (off : Nat) (active : Bool) :=
  materialize isa routing (if active then (u.mem r off).read .real else .zero)
def forwardInput (isa : Instructions) (routing : Routing) (s : Shape) (f : Flags) (bh iv : Nat) (u : State Node) :
    KernelReference.ForwardInput s Node where
  key := fun t i => read isa routing u "k" (keyAddress s bh t i.val) (decide (i.val<s.K))
  query := fun t i => read isa routing u "q" (keyAddress s bh t i.val) (decide (i.val<s.K))
  value := fun t j => read isa routing u "v" (valAddress s bh iv s.FBV t j.val) (decide (iv*s.FBV+j.val<s.V))
  beta := fun t j => if f.vectorBeta then
      read isa routing u "beta" (valAddress s bh iv s.FBV t j.val) (decide (iv*s.FBV+j.val<s.V))
    else read isa routing u "beta" (betaAddress s bh t) Bool.true
  seed := fun ji => if f.initial then .add .zero
    (read isa routing u "h0" (stateAddress s bh iv s.FBV ji.2.1.val ji.1.val)
      (decide (ji.2.1.val<s.K) && decide (iv*s.FBV+ji.1.val<s.V))) else .zero
def keySum (routing : Routing) (s : Shape) (values : ForwardTile s Node) (j : Fin s.FBV) :=
  reduce routing (shape:=[s.FBV,s.BK]) ⟨1,by simp⟩ Bool.false values (j,PUnit.unit)
def forward (routing : Routing) (s : Shape) (a : KernelReference.ForwardInput s Node) : Nat → ForwardTile s Node
  | 0 => a.seed
  | t+1 => fun ji =>
    let h := forward routing s a t
    let r := Node.sub (a.value t ji.1) (keySum routing s (fun vi => .mul (h vi) (a.key t vi.2.1)) ji.1)
    .add (h ji) (.mul (a.key t ji.2.1) (.mul r (a.beta t ji.1)))
def residual (routing : Routing) (s : Shape) (a : KernelReference.ForwardInput s Node) (t : Nat) (j : Fin s.FBV) :=
  Node.sub (a.value t j) (keySum routing s (fun ji => .mul (forward routing s a t ji) (a.key t ji.2.1)) j)
def output (routing : Routing) (s : Shape) (a : KernelReference.ForwardInput s Node) (scale : Word) (t : Nat) (j : Fin s.FBV) :=
  keySum routing s (fun ji => .mul (forward routing s a (t+1) ji) (.mul (a.query t ji.2.1) (.scale scale))) j

theorem input_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) :
    KernelReference.forwardInput (algebra isa routing scale) s f bh iv u=forwardInput isa routing s f bh iv u := by
  simp [KernelReference.forwardInput,KernelReference.read,KernelReference.cast,KernelReference.add,
    KernelReference.zero,forwardInput,read,algebra,scaleSlot]
theorem forward_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : KernelReference.ForwardInput s Node) (n : Nat) :
    KernelReference.forward (algebra isa routing scale) s a n=forward routing s a n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [KernelReference.forward,forward,ih]
    rfl
theorem residual_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : KernelReference.ForwardInput s Node) (t : Nat) (j : Fin s.FBV) :
    KernelReference.residual (algebra isa routing scale) s a t j=residual routing s a t j := by
  unfold KernelReference.residual
  rw [forward_lowering]
  rfl
theorem output_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : KernelReference.ForwardInput s Node) (t : Nat) (j : Fin s.FBV) :
    KernelReference.output (algebra isa routing scale) s a scaleSlot t j=output routing s a scale t j := by
  unfold KernelReference.output
  rw [forward_lowering]
  simp only [output,KernelReference.mul,KernelReference.scalar,
    KernelReference.keySum,keySum,algebra,ite_true]

#axiomsClean input_lowering
#axiomsClean forward_lowering
#axiomsClean residual_lowering
#axiomsClean output_lowering
#specNonCircular forward avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelReference.forward]
#specNonCircular output avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelReference.output]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeSpec
