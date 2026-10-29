import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelNativeSpec
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeBackwardSpec
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters KernelNativeCarrier
open KernelNativeSpec (read)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
abbrev Input := KernelBackwardReference.Input

def keySum (routing : Routing) (s : Shape) (x : Tile s Node) (j : Fin s.BBV) :=
  reduce routing (shape:=[s.BK,s.BBV]) ⟨0,by simp⟩ Bool.false x (j,PUnit.unit)
def valueSum (routing : Routing) (s : Shape) (x : Tile s Node) (i : Fin s.BK) :=
  reduce routing (shape:=[s.BK,s.BBV]) ⟨1,by simp⟩ Bool.false x (i,PUnit.unit)
def betaSum (routing : Routing) (s : Shape) (x : Fin s.BBV → Node) :=
  reduce routing (shape:=[s.BBV]) ⟨0,by simp⟩ Bool.false (fun j => x j.1) PUnit.unit
def input (isa : Instructions) (routing : Routing) (s : Shape) (f : Flags) (bh iv : Nat) (u : State Node) : Input s Node where
  key := fun t i => read isa routing u "k" (keyAddress s bh t i.val) (decide (i.val<s.K))
  query := fun t i => read isa routing u "q" (keyAddress s bh t i.val) (decide (i.val<s.K))
  value := fun t j => read isa routing u "v" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))
  beta := fun t j => if f.vectorBeta then
      read isa routing u "beta" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))
    else read isa routing u "beta" (betaAddress s bh t) Bool.true
  incoming := fun t j => read isa routing u "do" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))
  initial := fun ij => if f.initial then .add .zero
    (read isa routing u "h0" (stateAddress s bh iv s.BBV ij.1.val ij.2.1.val)
      (decide (ij.1.val<s.K) && decide (iv*s.BBV+ij.2.1.val<s.V))) else .zero
  terminal := fun ij => if f.terminalGrad then .add .zero
    (read isa routing u "dht" (stateAddress s bh iv s.BBV ij.1.val ij.2.1.val)
      (decide (ij.1.val<s.K) && decide (iv*s.BBV+ij.2.1.val<s.V))) else .zero
def enhance (s : Shape) (scale : Word) (a : Input s Node) (t : Nat) (h : Tile s Node) : Tile s Node :=
  fun ij => .add (h ij) (.mul (.mul (a.query t ij.1) (.scale scale)) (a.incoming t ij.2.1))
def rawDv (routing : Routing) (s : Shape) (a : Input s Node) (t : Nat) (g : Tile s Node) (j : Fin s.BBV) :=
  keySum routing s (fun ij => .mul (g ij) (a.key t ij.1)) j
def dv (routing : Routing) (s : Shape) (a : Input s Node) (t : Nat) (g : Tile s Node) (j : Fin s.BBV) :=
  Node.mul (rawDv routing s a t g j) (a.beta t j)
def dk (routing : Routing) (s : Shape) (a : Input s Node) (t : Nat) (g : Tile s Node) (i : Fin s.BK) :=
  valueSum routing s (fun ij => .mul (g ij) (.mul (a.value t ij.2.1) (a.beta t ij.2.1))) i
def dbeta (routing : Routing) (s : Shape) (a : Input s Node) (t : Nat) (g : Tile s Node) (j : Fin s.BBV) :=
  Node.mul (rawDv routing s a t g j) (a.value t j)
def retreat (routing : Routing) (s : Shape) (a : Input s Node) (t : Nat) (g : Tile s Node) : Tile s Node :=
  fun ij => .sub (g ij) (.mul (a.key t ij.1) (dv routing s a t g ij.2.1))
def reverse (routing : Routing) (s : Shape) (scale : Word) (a : Input s Node) : Nat → Tile s Node
  | 0 => a.terminal
  | n+1 => let t := s.T-1-n
    retreat routing s a t (enhance s scale a t (reverse routing s scale a n))
def forward (s : Shape) (a : Input s Node) : Nat → Tile s Node
  | 0 => a.initial
  | n+1 => fun ij => .add (forward s a n ij) (.mul (a.key n ij.1) (.mul (a.value n ij.2.1) (a.beta n ij.2.1)))
def correction (routing : Routing) (s : Shape) (h : Tile s Node)
    (dk : Fin s.BK → Node) (dv : Fin s.BBV → Node) (i : Fin s.BK) :=
  Node.sub (dk i) (valueSum routing s (fun ij => .mul (dv ij.2.1) (h ij)) i)
def dq (routing : Routing) (s : Shape) (scale : Word) (a : Input s Node) (t : Nat)
    (h : Tile s Node) (i : Fin s.BK) :=
  Node.mul (valueSum routing s (fun ij => .mul (h ij) (a.incoming t ij.2.1)) i) (.scale scale)

theorem input_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) :
    KernelBackwardReference.input (algebra isa routing scale) s f bh iv u=input isa routing s f bh iv u := by
  simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.add,
    KernelReference.zero,input,KernelNativeSpec.read,algebra,scaleSlot]
theorem enhance_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (t : Nat) (g : Tile s Node) :
    KernelBackwardReference.enhance (algebra isa routing scale) s scaleSlot a t g=enhance s scale a t g := by
  funext ij
  simp [KernelBackwardReference.enhance,KernelReference.add,KernelReference.mul,KernelReference.scalar,
    enhance,algebra]
theorem dv_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (t : Nat) (g : Tile s Node) :
    KernelBackwardReference.dv (algebra isa routing scale) s a t g=dv routing s a t g := rfl
theorem dk_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (t : Nat) (g : Tile s Node) :
    KernelBackwardReference.dk (algebra isa routing scale) s a t g=dk routing s a t g := rfl
theorem dbeta_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (t : Nat) (g : Tile s Node) :
    KernelBackwardReference.dbeta (algebra isa routing scale) s a t g=dbeta routing s a t g := rfl
theorem reverse_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (n : Nat) :
    KernelBackwardReference.reverse (algebra isa routing scale) s scaleSlot a n=reverse routing s scale a n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [KernelBackwardReference.reverse,reverse,ih,enhance_lowering]
    rfl
theorem forward_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (n : Nat) :
    KernelBackwardReference.forward (algebra isa routing scale) s a n=forward s a n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [KernelBackwardReference.forward,forward,ih]; rfl
theorem dq_lowering (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape)
    (a : Input s Node) (t : Nat) (g : Tile s Node) :
    KernelBackwardReference.dq (algebra isa routing scale) s scaleSlot a t g=dq routing s scale a t g := by
  funext i
  simp [KernelBackwardReference.dq,KernelReference.mul,KernelReference.scalar,
    KernelBackwardReference.valueSum,dq,valueSum,algebra]

#axiomsClean input_lowering
#axiomsClean enhance_lowering
#axiomsClean dv_lowering
#axiomsClean dk_lowering
#axiomsClean dbeta_lowering
#axiomsClean reverse_lowering
#axiomsClean forward_lowering
#axiomsClean dq_lowering
#specNonCircular reverse avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelBackwardReference.reverse]
#specNonCircular correction avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelBackwardReference.correction]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeBackwardSpec
