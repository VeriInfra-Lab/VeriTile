import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.flash_attn.Proofs.NativeResult
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section

def inputs (x : Layout) : List IOChannel :=
  [⟨.real,[x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim],"Q"⟩,
   ⟨.real,[x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim],"K"⟩,
   ⟨.real,[x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim],"V"⟩]
def outputs (x : Layout) : List IOChannel :=
  [⟨.real,[x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim],"O"⟩,
   ⟨.real,[x.shape.batches,x.shape.heads,x.shape.length],"L"⟩]
abbrev Inputs (A : Type) (x : Layout) := ChannelValues A (inputs x)
abbrev Outputs (A : Type) (x : Layout) := ChannelValues A (outputs x)

def inputStrides (x : Layout) (port : Fin 3) : Strides :=
  if port.val=0 then x.q else if port.val=1 then x.k else x.v

def memory (A : Type) (x : Layout) : MemoryIO A where
  inputs := inputs x
  outputs := outputs x
  scratch := []
  buffers := ["Q","K","V","O","L"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun i => FlashAttn.address x.q i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
    | ⟨1,_⟩ => fun i => FlashAttn.address x.k i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
    | ⟨2,_⟩ => fun i => FlashAttn.address x.v i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
    | ⟨k+3,hp⟩ => False.elim (by simp [inputs] at hp)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun i => FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
    | ⟨1,_⟩ => fun i => FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val
    | ⟨k+2,hp⟩ => False.elim (by simp [outputs] at hp)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0

/-- The original source consumes the runtime scalar register, including the
existing integer-one specialization and all raw FP32 words in the native model. -/
def invocation (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A) (job : Program) : Invocation A where
  kernel := ActualSource.kernel (parameters x)
  backend := FP.ProfiledRewrite.engine (withDot M dot)
  arguments := [⟨"sm_scale",.real,[],fun _ => scale⟩]
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then job.1 else if ax=1 then job.2 else 0
  numPids := some (fun ax => if ax=0 then (x.shape.length+127)/128
    else if ax=1 then x.shape.batches*x.shape.heads else 1)

def sourceIO (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
    (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A) :
    ProgramIO A B :=
  { memory A x with
    schedule := fun _ => (programs (parameters x)).map (invocation M dot x scale)
    encode := encode
    observe := observe }

/-- Zero extension is only a logical array convenience. The source's masked
padding still uses its own primitive zero before every retained cast. -/
def array {x : Layout} (xs : Inputs A x) (port : Fin 3) :
    TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim] → A :=
  match port with
  | ⟨0,_⟩ => xs (0 : Fin 3)
  | ⟨1,_⟩ => xs (1 : Fin 3)
  | ⟨2,_⟩ => xs (2 : Fin 3)
  | ⟨k+3,hp⟩ => False.elim (by omega)

def inputAt [Inhabited A] {x : Layout} (xs : Inputs A x) (port : Fin 3) (b h row col : Nat) : A :=
  if hb : b<x.shape.batches then
    if hh : h<x.shape.heads then
      if hr : row<x.shape.length then
        if hc : col<x.shape.dim then array xs port (⟨b,hb⟩,⟨h,hh⟩,⟨row,hr⟩,⟨col,hc⟩,()) else default
      else default
    else default
  else default

end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.LogicalIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section

abbrev Tensor (A : Type) (x : Layout) := Fin x.shape.batches → Fin x.shape.heads →
  Fin x.shape.length → Fin x.shape.dim → A
structure Inputs (A : Type) (x : Layout) where
  query : Tensor A x
  key : Tensor A x
  value : Tensor A x
structure Outputs (A : Type) (x : Layout) where
  values : Tensor A x
  logMass : Fin x.shape.batches → Fin x.shape.heads → Fin x.shape.length → A

def inputs (A : Type) (x : Layout) : PublicIO.Inputs A x ≃ Inputs A x where
  toFun v := ⟨fun b h r d => v (0 : Fin 3) (b,h,r,d,()),
    fun b h r d => v (1 : Fin 3) (b,h,r,d,()),fun b h r d => v (2 : Fin 3) (b,h,r,d,())⟩
  invFun v := Fin.cases (fun i => v.query i.1 i.2.1 i.2.2.1 i.2.2.2.1)
    (Fin.cases (fun i => v.key i.1 i.2.1 i.2.2.1 i.2.2.2.1)
      (Fin.cases (fun i => v.value i.1 i.2.1 i.2.2.1 i.2.2.2.1) (fun i => i.elim0)))
  left_inv := by intro v; funext port i; fin_cases port <;> cases i <;> rfl
  right_inv := by intro v; cases v; rfl

def outputs (A : Type) (x : Layout) : PublicIO.Outputs A x ≃ Outputs A x where
  toFun v := ⟨fun b h r d => v (0 : Fin 2) (b,h,r,d,()),fun b h r => v (1 : Fin 2) (b,h,r,())⟩
  invFun v := Fin.cases (fun i => v.values i.1 i.2.1 i.2.2.1 i.2.2.2.1)
    (Fin.cases (fun i => v.logMass i.1 i.2.1 i.2.2.1) (fun i => i.elim0))
  left_inv := by intro v; funext port i; fin_cases port <;> cases i <;> rfl
  right_inv := by intro v; cases v; rfl

def view (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A) :
    LogicalProgramIO A A (Inputs A x) (Outputs A x) where
  core := PublicIO.sourceIO M dot id x scale (fun _ _ => id)
  inputs := inputs A x
  outputs := outputs A x

/-- Logical-coordinate extension; undefined Option observations stay explicit. -/
def realAt {x : Layout} (v : Tensor (Option ℝ) x) (b h row col : Nat) : ℝ :=
  (if hb : b<x.shape.batches then if hh : h<x.shape.heads then
    if hr : row<x.shape.length then if hc : col<x.shape.dim then
      v ⟨b,hb⟩ ⟨h,hh⟩ ⟨row,hr⟩ ⟨col,hc⟩ else none else none else none else none).getD 0

def RealInputs (x : Layout) (xs : Inputs (Option ℝ) x) : Prop :=
  (∀ b h r d, ∃ value : ℝ,xs.query b h r d=some value) ∧
  (∀ b h r d, ∃ value : ℝ,xs.key b h r d=some value) ∧
  (∀ b h r d, ∃ value : ℝ,xs.value b h r d=some value)
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.LogicalIO
