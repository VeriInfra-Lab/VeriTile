import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.KernelArithmetic
import VeriTile.Triton.Float.ProgramIO

/-! Finite logical IO for the selected FP32 source. Equal region names express
in-place scaling; distinct names express separate input/output buffers. The
schedule retains every caller-selected program order. Physical address aliases
and the original finite integer lowering are connected in Proofs/PhysicalIO. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def parameters (src dst : RegionName) (n B : Nat) : Parameters :=
  ⟨src,src,dst,n,B⟩

abbrev inputChannels (src : RegionName) (n : Nat) : List IOChannel :=
  [⟨.real,[n],src⟩]
abbrev outputChannels (dst : RegionName) (n : Nat) : List IOChannel :=
  [⟨.real,[n],dst⟩]

/-- The typed source fixes FP32 storage/load/multiply/store; `.real` is its
Structural carrier tag, not an assertion that native arithmetic is exact. -/
def invocation (M : Algebra α) (src dst : RegionName) (n B pid : Nat) : Invocation α where
  kernel := candidateKernel src dst n B
  backend := M
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)

def io (M : Algebra α) (src dst : RegionName) (n B : Nat)
    (order : List Nat) : ProgramIO α α where
  inputs := inputChannels src n
  outputs := outputChannels dst n
  buffers := [src,dst]
  scratch := []
  read := fun i _ => by
    have hi : i=0 := Fin.eq_zero i
    subst i
    exact fun j => j.1.val
  inputMask := fun _ _ _ => True
  write := fun o _ => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact fun j => j.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => order.map (invocation M src dst n B)
  encode := id
  observe := fun _ _ => id

/-- Independent elementwise target; it never inspects source execution. -/
def formula (f : α → α) (src dst : RegionName) (n : Nat)
    (xs : ChannelValues α (inputChannels src n)) :
    ChannelValues α (outputChannels dst n) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact fun j => f (xs 0 j)

end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Interface

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Interface
open VeriTile.Triton FP.Structural
noncomputable section

/-- The complete sole vector input in ordinary logical coordinates. -/
def vectorInputs (src : RegionName) (n : Nat) :
    ChannelValues ℝ (inputChannels src n) ≃ (Fin n → ℝ) where
  toFun := fun xs i => xs 0 (i,PUnit.unit)
  invFun := fun xs => Fin.cases (fun j => xs j.1) (fun port => Fin.elim0 port)
  left_inv := by
    intro xs
    funext port j
    fin_cases port
    rcases j with ⟨i,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

/-- Every output cell is precisely one logical vector coordinate. -/
def vectorOutputs (dst : RegionName) (n : Nat) :
    ChannelValues ℝ (outputChannels dst n) ≃ (Fin n → ℝ) where
  toFun := fun ys i => ys 0 (i,PUnit.unit)
  invFun := fun ys => Fin.cases (fun j => ys j.1) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨i,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Interface

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Interface
open VeriTile.Triton FP.Structural

noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (M : Algebra ℝ) (src dst : RegionName) (n B : Nat) (order : List Nat) : LogicalProgramIO ℝ ℝ
    (Fin n → ℝ) (Fin n → ℝ) where
  core := io M src dst n B order
  inputs := vectorInputs src n
  outputs := vectorOutputs dst n

end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Interface
