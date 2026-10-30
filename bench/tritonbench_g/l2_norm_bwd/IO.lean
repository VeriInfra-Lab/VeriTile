import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.tritonbench_g.l2_norm_bwd.Kernels
import VeriTile.Triton.Float.ProgramIO
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

/-! Complete strided X/DY/DX IO. Epsilon is a runtime FP32 register argument;
the source precision is fixed by Kernels.lean, while `.real` is its erased
Structural floating payload channel.

This shared kernel interface describes the logical matrix cells, their strided
memory addresses and one invocation per row. Correct uses it to state the Real
formula; FPEquiv uses the same interface for the two actual source programs.
It adds no host or returned-Tensor proof obligation. -/
namespace VeriTile.Bench.Repairs.L2NormBwd.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (rows N : Nat) : List IOChannel :=
  [⟨.real,[rows,N],"X"⟩,⟨.real,[rows,N],"DY"⟩]
abbrev Inputs (A : Type) (rows N : Nat) := ChannelValues A (inputs rows N)

def rowValues [Inhabited A] {rows N : Nat} (xs : Inputs A rows N)
    (gradient : Bool) (row : Fin rows) (col : Nat) : A :=
  if hc : col<N then
    if gradient then xs ⟨1,by simp [inputs]⟩ (row,⟨col,hc⟩,PUnit.unit)
    else xs ⟨0,by simp [inputs]⟩ (row,⟨col,hc⟩,PUnit.unit)
  else default

def result (rows N : Nat) (f : Fin rows → Fin N → A) :
    ChannelValues A [⟨.real,[rows,N],"DX"⟩] :=
  Fin.cases (fun j => f j.1 j.2.1) (fun k => k.elim0)

/-- Supply the value of `tl.param eps : tl.float32` and select the row. -/
def invocation (M : Algebra A) (layout : Strides) (N B row : Nat) (eps : A) : Invocation A where
  kernel := source layout N B
  backend := M
  arguments := [⟨"eps",.real,[],fun _ => eps⟩]
  pids := fun ax => if ax=0 then row else 0

def schedule (M : Algebra A) (layout : Strides) (rows N B : Nat) (eps : A) : List (Invocation A) :=
  (List.range rows).map (fun row => invocation M layout N B row eps)

/-- Injective output coordinates are the existing source store-uniqueness
requirement. Input strides may overlap, including zero or repeated reads. -/
def OutputUnique (layout : Strides) (rows N : Nat) : Prop :=
  ∀ row col row' col',row<rows → col<N → row'<rows → col'<N →
    row*layout.dxRow+col*layout.dxCol=row'*layout.dxRow+col'*layout.dxCol →
    row=row' ∧ col=col'

def sourceIO (M : Algebra B) (encode : A → B) (observe : Nat → B → A)
    (layout : Strides) (rows N block : Nat) (eps : B) : ProgramIO A B where
  inputs := inputs rows N
  outputs := [⟨.real,[rows,N],"DX"⟩]
  scratch := []
  buffers := ["X","DY","DX"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun j => j.1.val*layout.xRow+j.2.1.val*layout.xCol
    | ⟨1,_⟩ => fun j => j.1.val*layout.dyRow+j.2.1.val*layout.dyCol
    | ⟨k+2,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*layout.dxRow+j.2.1.val*layout.dxCol) (fun k => k.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => schedule M layout rows N block eps
  encode := encode
  observe := Fin.cases (fun j => observe j.2.1.val) (fun k => k.elim0)

end
end VeriTile.Bench.Repairs.L2NormBwd.PublicIO

namespace VeriTile.Bench.Repairs.L2NormBwd.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- The complete repaired FP32 row grid, with independent X/DY/DX strides
and the actual runtime epsilon register. Payloads are never projected to Real. -/
abbrev originalIO {A : Type} (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (eps : A) : ProgramIO A A :=
  PublicIO.sourceIO M id (fun _ => id) layout rows N block eps

def candidateInvocation {A : Type} (M : Algebra A) (layout : Strides)
    (N block row : Nat) (eps : A) : Invocation A :=
  { PublicIO.invocation M layout N block row eps with
    kernel := L2NormBwd.indexCommuted layout N block }

def candidateSchedule {A : Type} (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (eps : A) : List (Invocation A) :=
  (List.range rows).map (fun row => candidateInvocation M layout N block row eps)

/-- Same channels, windows, masks, encoding and observer; only the actual
invoked source changes to the row-index-commuted candidate. -/
abbrev candidateIO {A : Type} (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (eps : A) :
    ProgramIO.Variant (originalIO M layout rows N block eps) where
  schedule := fun _ => candidateSchedule M layout rows N block eps
  observe := (originalIO M layout rows N block eps).observe

abbrev original {A : Type} [Inhabited A] (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (eps : A) :=
  ProgramIO.Implementation.original (originalIO M layout rows N block eps) (fun _ => True)

abbrev candidate {A : Type} [Inhabited A] (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (eps : A) :=
  ProgramIO.Implementation.ofVariant (candidateIO M layout rows N block eps) (fun _ => True)

end
end VeriTile.Bench.Repairs.L2NormBwd.FPEquiv

namespace VeriTile.Bench.Repairs.L2NormBwd.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (rows N : Nat) where
  x : Fin rows → Fin N → ℝ
  dy : Fin rows → Fin N → ℝ

def inputs (A : Algebra ℝ) (layout : Strides) (rows N B : Nat) (eps : ℝ) : (PublicIO.sourceIO A id (fun _ => id) layout rows N B eps).Inputs ≃ (Inputs rows N) where
  toFun := fun v => ⟨fun row col => v (0 : Fin 2) (row,col,()), fun row col => v (1 : Fin 2) (row,col,())⟩
  invFun := fun x i => match i with
    | ⟨0,_⟩ => fun j => x.x j.1 j.2.1
    | ⟨1,_⟩ => fun j => x.dy j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> rfl
  right_inv := by
    intro x
    cases x; rfl

def outputs (A : Algebra ℝ) (layout : Strides) (rows N B : Nat) (eps : ℝ) : (PublicIO.sourceIO A id (fun _ => id) layout rows N B eps).Outputs ≃ (Fin rows → Fin N → ℝ) where
  toFun := fun v row col => v (0 : Fin 1) (row,col,())
  invFun := fun y ⟨0,_⟩ j => y j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i
    rfl
  right_inv := fun _ => rfl

def io (A : Algebra ℝ) (layout : Strides) (rows N B : Nat) (eps : ℝ) : LogicalProgramIO ℝ ℝ (Inputs rows N) (Fin rows → Fin N → ℝ) where
  core := PublicIO.sourceIO A id (fun _ => id) layout rows N B eps
  inputs := inputs A layout rows N B eps
  outputs := outputs A layout rows N B eps

end
end VeriTile.Bench.Repairs.L2NormBwd.LogicalIO
