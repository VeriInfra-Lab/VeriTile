import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardRegisters
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelReference
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardReference
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open KernelReference (zero scalar add sub mul cast read stored)
noncomputable section
variable {α : Type} [Inhabited α]

def keySum (M : Algebra α) (s : Shape) (x : Tile s α) (j : Fin s.BBV) :=
  M.reduceSum (some .fp32) (shape:=[s.BK,s.BBV]) ⟨0,by simp⟩ Bool.false x (j,PUnit.unit)
def valueSum (M : Algebra α) (s : Shape) (x : Tile s α) (i : Fin s.BK) :=
  M.reduceSum (some .fp32) (shape:=[s.BK,s.BBV]) ⟨1,by simp⟩ Bool.false x (i,PUnit.unit)
def betaSum (M : Algebra α) (s : Shape) (x : Fin s.BBV → α) :=
  M.reduceSum (some .fp32) (shape:=[s.BBV]) ⟨0,by simp⟩ Bool.false (fun j => x j.1) PUnit.unit

structure Input (s : Shape) (α : Type) where
  key : Nat → Fin s.BK → α
  query : Nat → Fin s.BK → α
  value : Nat → Fin s.BBV → α
  beta : Nat → Fin s.BBV → α
  incoming : Nat → Fin s.BBV → α
  initial : Tile s α
  terminal : Tile s α

def input (M : Algebra α) (s : Shape) (f : Flags) (bh iv : Nat) (u : State α) : Input s α where
  key := fun t i => read M u "k" (keyAddress s bh t i.val) (decide (i.val<s.K))
  query := fun t i => read M u "q" (keyAddress s bh t i.val) (decide (i.val<s.K))
  value := fun t j => read M u "v" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))
  beta := fun t j => if f.vectorBeta then
      read M u "beta" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))
    else read M u "beta" (betaAddress s bh t) Bool.true
  incoming := fun t j => read M u "do" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))
  initial := fun ij => if f.initial then add M (zero M)
    (read M u "h0" (stateAddress s bh iv s.BBV ij.1.val ij.2.1.val)
      (decide (ij.1.val<s.K) && decide (iv*s.BBV+ij.2.1.val<s.V))) else zero M
  terminal := fun ij => if f.terminalGrad then add M (zero M)
    (read M u "dht" (stateAddress s bh iv s.BBV ij.1.val ij.2.1.val)
      (decide (ij.1.val<s.K) && decide (iv*s.BBV+ij.2.1.val<s.V))) else zero M

def enhance (M : Algebra α) (s : Shape) (scale : ℝ) (a : Input s α) (t : Nat) (h : Tile s α) : Tile s α :=
  fun ij => add M (h ij) (mul M (mul M (a.query t ij.1) (scalar M scale)) (a.incoming t ij.2.1))
def rawDv (M : Algebra α) (s : Shape) (a : Input s α) (t : Nat) (g : Tile s α) (j : Fin s.BBV) :=
  keySum M s (fun ij => mul M (g ij) (a.key t ij.1)) j
def dv (M : Algebra α) (s : Shape) (a : Input s α) (t : Nat) (g : Tile s α) (j : Fin s.BBV) :=
  mul M (rawDv M s a t g j) (a.beta t j)
def dk (M : Algebra α) (s : Shape) (a : Input s α) (t : Nat) (g : Tile s α) (i : Fin s.BK) :=
  valueSum M s (fun ij => mul M (g ij) (mul M (a.value t ij.2.1) (a.beta t ij.2.1))) i
def dbeta (M : Algebra α) (s : Shape) (a : Input s α) (t : Nat) (g : Tile s α) (j : Fin s.BBV) :=
  mul M (rawDv M s a t g j) (a.value t j)
def retreat (M : Algebra α) (s : Shape) (a : Input s α) (t : Nat) (g : Tile s α) : Tile s α :=
  fun ij => sub M (g ij) (mul M (a.key t ij.1) (dv M s a t g ij.2.1))
def reverse (M : Algebra α) (s : Shape) (scale : ℝ) (a : Input s α) : Nat → Tile s α
  | 0 => a.terminal
  | n+1 => let t := s.T-1-n
      retreat M s a t (enhance M s scale a t (reverse M s scale a n))

def forward (M : Algebra α) (s : Shape) (a : Input s α) : Nat → Tile s α
  | 0 => a.initial
  | n+1 => fun ij => add M (forward M s a n ij)
      (mul M (a.key n ij.1) (mul M (a.value n ij.2.1) (a.beta n ij.2.1)))
def correction (M : Algebra α) (s : Shape) (h : Tile s α)
    (dk : Fin s.BK → α) (dv : Fin s.BBV → α) (i : Fin s.BK) :=
  sub M (dk i) (valueSum M s (fun ij => mul M (dv ij.2.1) (h ij)) i)
def dq (M : Algebra α) (s : Shape) (scale : ℝ) (a : Input s α) (t : Nat)
    (h : Tile s α) (i : Fin s.BK) :=
  mul M (valueSum M s (fun ij => mul M (h ij) (a.incoming t ij.2.1)) i) (scalar M scale)

#specNonCircular reverse avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
#specNonCircular forward avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
#specNonCircular correction avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardReference
