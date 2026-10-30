import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Memory.LogicalIO

/-! Logical tensor coordinates for the existing typed ProgramIO contract.
No memory layout, schedule, observer or execution semantics is duplicated. -/
namespace VeriTile.Triton.FP.Structural

structure LogicalProgramIO (α β Inputs Outputs : Type) where
  core : ProgramIO α β
  inputs : core.Inputs ≃ Inputs
  outputs : core.Outputs ≃ Outputs

namespace LogicalProgramIO
variable {α β I O : Type}

def lift (io : LogicalProgramIO α β I O) (f : I → O) : io.core.Inputs → io.core.Outputs :=
  fun raw => io.outputs.symm (f (io.inputs raw))

def Implements [Inhabited β] (io : LogicalProgramIO α β I O) (f : I → O) : Prop :=
  io.core.Implements (io.lift f)

def ImplementsOn [Inhabited β] (io : LogicalProgramIO α β I O)
    (domain : I → Prop) (f : I → O) : Prop :=
  io.core.ImplementsOn (fun raw => domain (io.inputs raw)) (io.lift f)

scoped infix:25 " ⊨ " => LogicalProgramIO.Implements
scoped notation:25 io " ⊨{" domain "} " f => LogicalProgramIO.ImplementsOn io domain f

theorem implements_iff [Inhabited β] (io : LogicalProgramIO α β I O) (f : I → O) :
    (io ⊨ f) ↔ io.core.Implements (io.lift f) := Iff.rfl

theorem implementsOn_iff [Inhabited β] (io : LogicalProgramIO α β I O)
    (domain : I → Prop) (f : I → O) :
    (io ⊨{domain} f) ↔ io.core.ImplementsOn
      (fun raw => domain (io.inputs raw)) (io.lift f) := Iff.rfl
end LogicalProgramIO

namespace ChannelValues

/-- Split a typed input/output channel list without dropping its tail. -/
def consEquiv (α : Type) (port : IOChannel) (ports : List IOChannel) :
    ChannelValues α (port :: ports) ≃
      (Values α port.dtype port.shape × ChannelValues α ports) where
  toFun := fun vals => (vals 0, fun i => vals i.succ)
  invFun := fun vals => Fin.cases vals.1 vals.2
  left_inv := by
    intro vals
    funext i
    refine Fin.cases ?_ (fun _ => ?_) i <;> rfl
  right_inv := by
    rintro ⟨head, tail⟩
    rfl

/-- Remove a sole channel wrapper; all of its typed tensor values remain. -/
def singleEquiv (α : Type) (port : IOChannel) :
    ChannelValues α [port] ≃ Values α port.dtype port.shape where
  toFun := fun vals => vals 0
  invFun := fun vals => Fin.cases vals (fun i => Fin.elim0 i)
  left_inv := by
    intro vals
    funext i
    exact Fin.cases rfl (fun i => Fin.elim0 i) i
  right_inv := fun _ => rfl

end ChannelValues

/-- Curried tensor coordinates retain the source's scalar dtype. -/
def CurriedValues (α : Type) (dtype : TileDType) : TileShape → Type
  | [] => Value α dtype
  | n :: shape => Fin n → CurriedValues α dtype shape

namespace Values

/-- Remove only the tuple/PUnit index encoding, for any tensor shape. -/
def curryEquiv (α : Type) (dtype : TileDType) (shape : TileShape) :
    Values α dtype shape ≃ CurriedValues α dtype shape :=
  match shape with
  | [] => {
      toFun := fun vals => vals PUnit.unit
      invFun := fun val _ => val
      left_inv := by intro vals; funext i; cases i; rfl
      right_inv := fun _ => rfl }
  | n :: tail => {
      toFun := fun vals i => curryEquiv α dtype tail (fun j => vals (i, j))
      invFun := fun vals ij => (curryEquiv α dtype tail).symm (vals ij.1) ij.2
      left_inv := by
        intro vals
        funext ij
        exact congrFun ((curryEquiv α dtype tail).symm_apply_apply
          (fun j => vals (ij.1, j))) ij.2
      right_inv := by
        intro vals
        funext i
        exact (curryEquiv α dtype tail).apply_symm_apply (vals i) }

def scalarEquiv (α : Type) (dtype : TileDType) := curryEquiv α dtype []
def vectorEquiv (α : Type) (dtype : TileDType) (n : Nat) := curryEquiv α dtype [n]
def matrixEquiv (α : Type) (dtype : TileDType) (m n : Nat) := curryEquiv α dtype [m, n]
def tensor3Equiv (α : Type) (dtype : TileDType) (l m n : Nat) := curryEquiv α dtype [l, m, n]
def tensor4Equiv (α : Type) (dtype : TileDType) (k l m n : Nat) := curryEquiv α dtype [k, l, m, n]

end Values
end VeriTile.Triton.FP.Structural
