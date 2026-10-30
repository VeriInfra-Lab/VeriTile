import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.index_select_bwd.Proofs.Kernel
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! IO of the actual FP32 slice kernel. The destination is an input and the
single output, so earlier duplicate-row contributions remain visible. Source
FP32 precision is retained in `kernel`; `.real` is its Structural cell type. -/
namespace VeriTile.Bench.Repairs.IndexSelectBwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (p : Launch) : List IOChannel :=
  [⟨.real, [p.block], p.destination⟩, ⟨.real, [p.block], p.incoming⟩]
def outputs (p : Launch) : List IOChannel := [⟨.real, [p.block], p.destination⟩]
abbrev Inputs (A : Type) (p : Launch) := ChannelValues A (inputs p)
abbrev Outputs (A : Type) (p : Launch) := ChannelValues A (outputs p)

def inputLane (p : Launch) (i : Fin (inputs p).length)
    : TileIndex (inputs p)[i].shape → Fin p.block :=
  match i with
  | ⟨0, _⟩ => fun j => j.1
  | ⟨1, _⟩ => fun j => j.1
  | ⟨n + 2, h⟩ => False.elim (by simp [inputs] at h)

def outputLane (p : Launch) (o : Fin (outputs p).length)
    : TileIndex (outputs p)[o].shape → Fin p.block :=
  match o with
  | ⟨0, _⟩ => fun j => j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (p : Launch) : MemoryIO A where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.destination, p.incoming]
  read := fun i _ j => (inputLane p i j).val
  inputMask := fun i _ j => (inputLane p i j).val < p.count
  write := fun o _ j => (outputLane p o j).val
  outputMask := fun o _ j => (outputLane p o j).val < p.count
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def invocation {A : Type} (rest : Algebra A) (add : A → A → A) (p : Launch) : Invocation A where
  kernel := kernel p
  backend := Source.backend (KernelProof.arithmetic rest add)
  pids := fun _ => 0

def programIO {A : Type} (rest : Algebra A) (add : A → A → A) (p : Launch) : ProgramIO A A :=
  { memory A p with
    schedule := fun _ => [invocation rest add p]
    encode := id
    observe := fun _ _ => id }

def current {A : Type} {p : Launch} (values : Inputs A p) (j : Fin p.block) : A :=
  values ⟨0, by simp [inputs]⟩ (j, PUnit.unit)
def incoming {A : Type} {p : Launch} (values : Inputs A p) (j : Fin p.block) : A :=
  values ⟨1, by simp [inputs]⟩ (j, PUnit.unit)

def result {A : Type} {p : Launch} (value : Fin p.block → A) : Outputs A p :=
  fun o => match o with
  | ⟨0, _⟩ => fun j => value j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.IndexSelectBwd.Interface

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Interface
open VeriTile Triton FP.Structural
noncomputable section

def adaptiveInvocation {A : Type} (rest : Algebra A) (add : A → A → A) (p : Launch) :
    Invocation A := invocation rest add {p with block := tailBlock p.count}

def adaptiveVariant {A : Type} (rest : Algebra A) (add : A → A → A) (p : Launch) :
    ProgramIO.Variant (programIO rest add p) where
  schedule := fun _ => [adaptiveInvocation rest add p]
  observe := fun _ _ => id

def original {A : Type} (rest : Algebra A) (add : A → A → A) (p : Launch) :=
  ProgramIO.Implementation.original (programIO rest add p) (fun _ => True)

def adaptive {A : Type} (rest : Algebra A) (add : A → A → A) (p : Launch) :=
  ProgramIO.Implementation.ofVariant (adaptiveVariant rest add p) (fun _ => True)
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Interface

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Both slice inputs, including the current value after earlier duplicates. -/
structure SliceValues (A : Type) (p : Launch) where
  current : Fin p.block → A
  incoming : Fin p.block → A

def inputCoordinates (A : Type) (p : Launch) : Inputs A p ≃ SliceValues A p where
  toFun := fun xs => ⟨current xs, incoming xs⟩
  invFun := fun xs => Fin.cases (fun j => xs.current j.1)
    (Fin.cases (fun j => xs.incoming j.1) (fun i => Fin.elim0 i))
  left_inv := by
    intro xs
    funext i j
    fin_cases i <;> cases j with | mk a b => cases b; rfl
  right_inv := by intro xs; cases xs; rfl

def outputCoordinates (A : Type) (p : Launch) : Outputs A p ≃ (Fin p.block → A) where
  toFun := fun xs i => xs ⟨0, by simp [outputs]⟩ (i, ())
  invFun := result
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    cases j with | mk a b => cases b; rfl
  right_inv := fun _ => rfl

def logicalIO (rest : Algebra A) (add : A → A → A) (p : Launch) :
    LogicalProgramIO A A (SliceValues A p) (Fin p.block → A) where
  core := programIO rest add p
  inputs := inputCoordinates A p
  outputs := outputCoordinates A p

end
end VeriTile.Bench.Repairs.IndexSelectBwd.Interface
