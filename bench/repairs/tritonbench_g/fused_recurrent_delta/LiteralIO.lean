import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardReference
import VeriTile.Triton.Float.ObservedProgramIO

/-! Historical literal-reference IO used only to reuse the established
proof foundation. Actual runtime IO is RuntimeIO.io in IO.lean.

Finite logical channels for each complete original forward/backward program.
The source retains FP32 loads, operations and stores. `real` is the Structural
carrier, not an assertion that native words compute real arithmetic. The two
beta views share one region and have mutually exclusive masks. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section

def forwardInputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.T,s.BK],"q"⟩,⟨.real,[s.T,s.BK],"k"⟩,
   ⟨.real,[s.T,s.FBV],"v"⟩,⟨.real,[s.T,s.FBV],"beta"⟩,
   ⟨.real,[s.T],"beta"⟩,⟨.real,[s.BK,s.FBV],"h0"⟩]
def forwardOutputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.T,s.FBV],"o"⟩,⟨.real,[s.T,s.FBV],"v"⟩,
   ⟨.real,[s.BK,s.FBV],"ht"⟩]
def backwardInputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.T,s.BK],"q"⟩,⟨.real,[s.T,s.BK],"k"⟩,
   ⟨.real,[s.T,s.BBV],"v"⟩,⟨.real,[s.T,s.BBV],"beta"⟩,
   ⟨.real,[s.T],"beta"⟩,⟨.real,[s.T,s.BBV],"do"⟩,
   ⟨.real,[s.BK,s.BBV],"h0"⟩,⟨.real,[s.BK,s.BBV],"dht"⟩]
def backwardOutputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.T,s.BK],"dq"⟩,⟨.real,[s.T,s.BK],"dk"⟩,
   ⟨.real,[s.T,s.BBV],"dv"⟩,⟨.real,[s.T,s.BBV],"dbeta"⟩,
   ⟨.real,[s.T],"dbeta"⟩,⟨.real,[s.BK,s.BBV],"dh0"⟩]
abbrev ForwardInputs (α : Type) (s : Shape) := ChannelValues α (forwardInputs s)
abbrev ForwardOutputs (α : Type) (s : Shape) := ChannelValues α (forwardOutputs s)
abbrev BackwardInputs (α : Type) (s : Shape) := ChannelValues α (backwardInputs s)
abbrev BackwardOutputs (α : Type) (s : Shape) := ChannelValues α (backwardOutputs s)

def job {α : Type} (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) : Invocation α where
  kernel := source kind s f scale
  backend := FP.ProfiledRewrite.engine M
  pids := fun ax => if ax=0 then iv else if ax=1 then 0 else bh
  setPids := fun ax => decide (ax<3)

def forwardIO {α : Type} (M : Algebra α) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) : ProgramIO α α where
  inputs := forwardInputs s
  outputs := forwardOutputs s
  scratch := []
  buffers := ["q","k","v","beta","o","h0","ht"]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => keyAddress s bh j.1.val j.2.1.val
    | ⟨2,_⟩ | ⟨3,_⟩ => fun j => valAddress s bh iv s.FBV j.1.val j.2.1.val
    | ⟨4,_⟩ => fun j => betaAddress s bh j.1.val
    | ⟨5,_⟩ => fun j => stateAddress s bh iv s.FBV j.1.val j.2.1.val
    | ⟨n+6,h⟩ => False.elim (by simp [forwardInputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.2.1.val<s.K
    | ⟨2,_⟩ => fun j => iv*s.FBV+j.2.1.val<s.V
    | ⟨3,_⟩ => fun j => f.vectorBeta=Bool.true ∧ iv*s.FBV+j.2.1.val<s.V
    | ⟨4,_⟩ => fun _ => f.vectorBeta=Bool.false
    | ⟨5,_⟩ => fun j => f.initial=Bool.true ∧ j.1.val<s.K ∧ iv*s.FBV+j.2.1.val<s.V
    | ⟨n+6,h⟩ => False.elim (by simp [forwardInputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => valAddress s bh iv s.FBV j.1.val j.2.1.val
    | ⟨2,_⟩ => fun j => stateAddress s bh iv s.FBV j.1.val j.2.1.val
    | ⟨n+3,h⟩ => False.elim (by simp [forwardOutputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => iv*s.FBV+j.2.1.val<s.V
    | ⟨2,_⟩ => fun j => f.final=Bool.true ∧ j.1.val<s.K ∧ iv*s.FBV+j.2.1.val<s.V
    | ⟨n+3,h⟩ => False.elim (by simp [forwardOutputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [job M .forward s f scale bh iv]
  encode := id
  observe := fun _ _ => id

def backwardIO {α : Type} (M : Algebra α) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) : ProgramIO α α where
  inputs := backwardInputs s
  outputs := backwardOutputs s
  scratch := []
  buffers := ["q","k","v","beta","do","h0","dht","dq","dk","dv","dbeta","dh0"]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => keyAddress s bh j.1.val j.2.1.val
    | ⟨2,_⟩ | ⟨3,_⟩ | ⟨5,_⟩ => fun j => valAddress s bh iv s.BBV j.1.val j.2.1.val
    | ⟨4,_⟩ => fun j => betaAddress s bh j.1.val
    | ⟨6,_⟩ | ⟨7,_⟩ => fun j => stateAddress s bh iv s.BBV j.1.val j.2.1.val
    | ⟨n+8,h⟩ => False.elim (by simp [backwardInputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.2.1.val<s.K
    | ⟨2,_⟩ | ⟨5,_⟩ => fun j => iv*s.BBV+j.2.1.val<s.V
    | ⟨3,_⟩ => fun j => f.vectorBeta=Bool.true ∧ iv*s.BBV+j.2.1.val<s.V
    | ⟨4,_⟩ => fun _ => f.vectorBeta=Bool.false
    | ⟨6,_⟩ => fun j => f.initial=Bool.true ∧ j.1.val<s.K ∧ iv*s.BBV+j.2.1.val<s.V
    | ⟨7,_⟩ => fun j => f.terminalGrad=Bool.true ∧ j.1.val<s.K ∧ iv*s.BBV+j.2.1.val<s.V
    | ⟨n+8,h⟩ => False.elim (by simp [backwardInputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => keyAddress s (plane s bh iv) j.1.val j.2.1.val
    | ⟨2,_⟩ => fun j => valAddress s bh iv s.BBV j.1.val j.2.1.val
    | ⟨3,_⟩ => fun j => valAddress s (plane s bh iv) iv s.BBV j.1.val j.2.1.val
    | ⟨4,_⟩ => fun j => plane s bh iv*s.T+j.1.val
    | ⟨5,_⟩ => fun j => stateAddress s bh iv s.BBV j.1.val j.2.1.val
    | ⟨n+6,h⟩ => False.elim (by simp [backwardOutputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.2.1.val<s.K
    | ⟨2,_⟩ => fun j => iv*s.BBV+j.2.1.val<s.V
    | ⟨3,_⟩ => fun j => f.vectorBeta=Bool.true ∧ iv*s.BBV+j.2.1.val<s.V
    | ⟨4,_⟩ => fun _ => f.vectorBeta=Bool.false
    | ⟨5,_⟩ => fun j => f.initialGrad=Bool.true ∧ j.1.val<s.K ∧ iv*s.BBV+j.2.1.val<s.V
    | ⟨n+6,h⟩ => False.elim (by simp [backwardOutputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [job M .backward s f scale bh iv]
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
