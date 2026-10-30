import bench.tritonbench_g.matmul_kernel.Proofs.KernelSafety
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.tritonbench_g.matmul_kernel.Proofs.KernelReal
import VeriTile.Triton.Float.LogicalProgramIO


namespace VeriTile.Bench.Repairs.MatmulKernel.PublicIO
open VeriTile.Triton FP.Structural KernelSource KernelScalar
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

def invocation (p : Config) (M : Algebra α) (D : DotAcc α) (pid : Program) : Invocation α where
  kernel := KernelSource.source p
  backend := KernelSource.backend M D
  pids := fun d => if d=0 then pid.1 else pid.2
  setPids := fun d => decide (d<2)

def inputs (p : Config) : List IOChannel :=
  [⟨.real,[p.M,p.K],"A"⟩,⟨.real,[p.K,p.N],"B"⟩]
def outputs (p : Config) : List IOChannel := [⟨.real,[p.M,p.N],"C"⟩]

def io (p : Config) (M : Algebra α) (D : DotAcc α) : ProgramIO α α where
  inputs := inputs p
  outputs := outputs p
  buffers := ["A","B","C"]
  scratch := []
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => p.A.addr lane.1.val lane.2.1.val
    | ⟨1,_⟩ => fun lane => p.B.addr lane.1.val lane.2.1.val
    | ⟨n+2,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => p.C.addr lane.1.val lane.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (programs p).map (invocation p M D)
  encode := id
  observe := fun _ _ => id

def arrayA (p : Config) (xs : ChannelValues α (inputs p)) (r k : Nat) : α :=
  if hr : r<p.M then if hk : k<p.K then
    xs ⟨0,by simp [inputs]⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) else default else default

def arrayB (p : Config) (xs : ChannelValues α (inputs p)) (k n : Nat) : α :=
  if hk : k<p.K then if hn : n<p.N then
    xs ⟨1,by simp [inputs]⟩ (⟨k,hk⟩,⟨n,hn⟩,PUnit.unit) else default else default

def expected (p : Config) (f : (Nat → Nat → α) → (Nat → Nat → α) → Nat → Nat → α)
    (xs : ChannelValues α (inputs p)) : ChannelValues α (outputs p) :=
  fun port => match port with
    | ⟨0,_⟩ => fun lane => f (arrayA p xs) (arrayB p xs) lane.1.val lane.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.MatmulKernel.PublicIO

namespace VeriTile.Bench.Repairs.MatmulKernel.Pair
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (p : Config) (M : Algebra α) (D : DotAcc α) (pid : Program) : Invocation α :=
  { PublicIO.invocation p M D pid with kernel := remaining p }

def candidateIO (p : Config) (M : Algebra α) (D : DotAcc α) : ProgramIO α α :=
  { PublicIO.io p M D with schedule := fun _ => (programs p).map (invocation p M D) }

end
end VeriTile.Bench.Repairs.MatmulKernel.Pair

namespace VeriTile.Bench.Repairs.MatmulKernel.PairIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

def variant (p : Config) (M : Algebra α) (D : DotAcc α) : ProgramIO.Variant (PublicIO.io p M D) where
  schedule := (Pair.candidateIO p M D).schedule
  observe := (PublicIO.io p M D).observe

end
end VeriTile.Bench.Repairs.MatmulKernel.PairIO



namespace VeriTile.Bench.Repairs.MatmulKernel.ReaderIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Both complete finite input matrices; no channel or unused value is discarded. -/
structure Inputs (p : Config) where
  A : Fin p.M → Fin p.K → ℝ
  B : Fin p.K → Fin p.N → ℝ

def inputs (p : Config) : ChannelValues ℝ (PublicIO.inputs p) ≃ Inputs p where
  toFun := fun xs => ⟨(fun r t => xs (0 : Fin 2) (r,t,())), (fun t c => xs (1 : Fin 2) (t,c,()))⟩
  invFun := fun xs => Fin.cases (fun j => xs.A j.1 j.2.1)
    (Fin.cases (fun j => xs.B j.1 j.2.1) (fun i => i.elim0))
  left_inv := by
    intro xs; funext i j; fin_cases i <;> rcases j with ⟨a,b,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputs (p : Config) : ChannelValues ℝ (PublicIO.outputs p) ≃
    (Fin p.M → Fin p.N → ℝ) where
  toFun := fun ys r c => ys (0 : Fin 1) (r,c,())
  invFun := fun ys => Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by
    intro ys; funext i j; fin_cases i; rcases j with ⟨a,b,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

/-- Only coordinate equivalences; the original typed IO, complete schedule,
input binding, successful execution, safety and exterior frame are reused. -/
def view (p : Config) :
    LogicalProgramIO ℝ ℝ (Inputs p) (Fin p.M → Fin p.N → ℝ) where
  core := PublicIO.io p KernelReal.scalar (KernelScalar.dotAcc KernelReal.dot)
  inputs := inputs p
  outputs := outputs p
end
end VeriTile.Bench.Repairs.MatmulKernel.ReaderIO
