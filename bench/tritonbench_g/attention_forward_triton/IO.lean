import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.attention_forward_triton.Proofs.HoistedSource
import bench.tritonbench_g.attention_forward_triton.Proofs.NativeReference
import bench.tritonbench_g.attention_forward_triton.Proofs.RealBackend
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source Whole
noncomputable section

abbrev DataLane (p : Parameters) := TileIndex [p.batches,p.heads,p.length,96]
abbrev ScaleLane (p : Parameters) (block : Nat) :=
  TileIndex [p.batches,p.heads,(p.length+block-1)/block]

/-- Q/K/V and both scale arrays have FP32 source storage. The Structural
backend reads their precision-erased `.real` channels; output stays FP16. -/
def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.real,[p.batches,p.heads,p.length,96],p.Q⟩,
   ⟨.real,[p.batches,p.heads,p.length,96],p.K⟩,
   ⟨.real,[p.batches,p.heads,p.length,96],p.V⟩,
   ⟨.real,[p.batches,p.heads,(p.length+127)/128],p.QS⟩,
   ⟨.real,[p.batches,p.heads,(p.length+63)/64],p.KS⟩]

def outputChannels (p : Parameters) : List IOChannel :=
  [⟨.fp16,[p.batches,p.heads,p.length,96],p.Out⟩]

abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputChannels p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputChannels p)

def dataAddress (st : Strides) (j : DataLane p) : Nat :=
  j.1.val*st.batch+j.2.1.val*st.head+j.2.2.1.val*st.row+j.2.2.2.1.val*st.dim

def scaleAddress (p : Parameters) (block : Nat) (j : ScaleLane p block) : Nat :=
  (j.1.val*p.heads+j.2.1.val)*((p.length+block-1)/block)+j.2.2.1.val

def inputAddress (p : Parameters) :
    (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Nat :=
  Fin.cases (dataAddress p.q)
    (Fin.cases (dataAddress p.k)
      (Fin.cases (dataAddress p.v)
        (Fin.cases (scaleAddress p 128)
          (Fin.cases (scaleAddress p 64) (fun i => Fin.elim0 i)))))

def payload (p : Parameters) (vals : Inputs A p) :
    (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → A :=
  Fin.cases (vals ⟨0,by simp [inputChannels]⟩)
    (Fin.cases (vals ⟨1,by simp [inputChannels]⟩)
      (Fin.cases (vals ⟨2,by simp [inputChannels]⟩)
        (Fin.cases (vals ⟨3,by simp [inputChannels]⟩)
          (Fin.cases (vals ⟨4,by simp [inputChannels]⟩) (fun i => Fin.elim0 i)))))

abbrev InputPosition (p : Parameters) :=
  (i : Fin (inputChannels p).length) × TileIndex (inputChannels p)[i].shape

/-- A total extension of the finite logical inputs. Overlapping input windows
share one memory value by the simultaneous input binding. No separation among
Q/K/V/scales is introduced. Values outside their windows are irrelevant. -/
def logicalRead (zero : A) (p : Parameters) (vals : Inputs A p) (r : RegionName) (off : Nat) : A := by
  classical
  exact if h : ∃ pos : InputPosition p,
      (inputChannels p)[pos.1].buf=r ∧ inputAddress p pos.1 pos.2=off then
    let pos := Classical.choose h
    payload p vals pos.1 pos.2
  else zero

def memory (p : Parameters) : MemoryIO A where
  inputs := inputChannels p
  outputs := outputChannels p
  buffers := [p.Q,p.K,p.V,p.QS,p.KS,p.Out]
  read := fun i _ => inputAddress p i
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ => dataAddress p.o) (fun i => Fin.elim0 i)
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (dot : Dot A) (p : Parameters) (pid : Whole.Program) : Invocation A where
  kernel := Source.kernel p
  backend := engine (withDot M dot)
  clearRegisters := Bool.true
  pids := ids pid
  numPids := some (dimensions (grid p))

def realIO (p : Parameters) (work : List Whole.Program) : ProgramIO Real.R Real.R where
  toMemoryIO := memory p
  schedule := fun _ => work.map (invocation Real.algebra Real.dotModel p)
  encode := id
  observe := fun _ _ => id

/-- Real inputs are finite reals, matching the original Real interpretation.
This is a mathematical representation condition, never an experimental law. -/
def RealInputs (p : Parameters) (vals : Inputs Real.R p) : Prop :=
  ∀ i j, (payload p vals i j).isSome

def mathematical (p : Parameters) (vals : Inputs Real.R p) : Outputs Real.R p :=
  Fin.cases (fun j => some (RealMath.expected p.length
    (fun n d => (logicalRead (some 0) p vals p.Q
      (j.1.val*p.q.batch+j.2.1.val*p.q.head+n*p.q.row+d*p.q.dim)).getD 0)
    (fun n d => (logicalRead (some 0) p vals p.K
      (j.1.val*p.k.batch+j.2.1.val*p.k.head+d*p.k.dim+n*p.k.row)).getD 0)
    (fun n d => (logicalRead (some 0) p vals p.V
      (j.1.val*p.v.batch+j.2.1.val*p.v.head+n*p.v.row+d*p.v.dim)).getD 0)
    (fun block => (logicalRead (some 0) p vals p.QS
      ((j.1.val*p.heads+j.2.1.val)*((p.length+127)/128)+block)).getD 0)
    (fun block => (logicalRead (some 0) p vals p.KS
      ((j.1.val*p.heads+j.2.1.val)*((p.length+63)/64)+block)).getD 0)
    j.2.2.1.val j.2.2.2.1.val)) (fun i => Fin.elim0 i)

def wordEncode (w : Native.Word) : Native.Expr := fun _ => .ready (.bits w)

def nativeIO (I : Native.Instructions) (p : Parameters) (work : List Whole.Program) :
    ProgramIO Native.Word Native.Expr where
  toMemoryIO := memory p
  schedule := fun _ => work.map (invocation Native.algebra Native.dotModel p)
  encode := wordEncode
  observe := Fin.cases (fun j value =>
    (Native.lower (value (Native.physicalOwner j.2.2.2.1.val))).eval I.ops) (fun i => Fin.elim0 i)

/-- Independent fixed-K8 TF32 / fixed-K16 FP16 instruction recurrence, with
original reduction trees, casts and replica ownership. All words are admitted. -/
def nativeTarget (I : Native.Instructions) (p : Parameters) (vals : Inputs Native.Word p) :
    Outputs Native.Word p :=
  Fin.cases (fun j =>
    let input : NativeReference.Inputs :=
      ⟨fun n d => logicalRead 0 p vals p.Q
          (j.1.val*p.q.batch+j.2.1.val*p.q.head+n*p.q.row+d*p.q.dim),
       fun n d => logicalRead 0 p vals p.K
          (j.1.val*p.k.batch+j.2.1.val*p.k.head+d*p.k.dim+n*p.k.row),
       fun n d => logicalRead 0 p vals p.V
          (j.1.val*p.v.batch+j.2.1.val*p.v.head+n*p.v.row+d*p.v.dim),
       fun block => logicalRead 0 p vals p.QS
          ((j.1.val*p.heads+j.2.1.val)*((p.length+127)/128)+block),
       fun block => logicalRead 0 p vals p.KS
          ((j.1.val*p.heads+j.2.1.val)*((p.length+63)/64)+block)⟩
    (NativeReference.attention p.length j.2.2.1.val input
      ⟨j.2.2.2.1.val,Nat.lt_trans j.2.2.2.1.isLt (by decide)⟩).eval I.ops)
    (fun i => Fin.elim0 i)

end
end VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedIO
open VeriTile Triton FP.Structural Source Whole
open AttentionIO
noncomputable section

def invocation (M : Algebra A) (dot : Dot A) (p : Parameters) (pid : Whole.Program) : Invocation A :=
  { AttentionIO.invocation M dot p pid with kernel := HoistedSource.kernel p }

def nativeIO (I : Native.Instructions) (p : Parameters) (work : List Whole.Program) :
    ProgramIO Native.Word Native.Expr :=
  { AttentionIO.nativeIO I p work with
    schedule := fun _ => work.map (invocation Native.algebra Native.dotModel p) }

end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedIO

/-! Logical array coordinates for the existing complete IO. The storage field
retains every original input cell, including aliases and padding; the view is
an equivalence and never replaces the underlying source, schedule or frame. -/
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.LogicalIO
open VeriTile Triton FP.Structural Source Whole AttentionIO
noncomputable section

structure Backing (p : Parameters) where
  storage : AttentionIO.Inputs Real.R p

abbrev Outputs (p : Parameters) :=
  Fin p.batches → Fin p.heads → Fin p.length → Fin (96) → Real.R

def outputsEquiv (p : Parameters) (work : List Program) :
    (AttentionIO.realIO p work).Outputs ≃ Outputs p where
  toFun := fun values b h row col => values (0 : Fin 1) (b,h,row,col,())
  invFun := fun values => Fin.cases
    (fun j => values j.1 j.2.1 j.2.2.1 j.2.2.2.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro values
    funext o j
    have zero : o = (0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨b,h,row,col,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl


def read (xs : Backing p) := (fun r off => (AttentionIO.logicalRead (some 0) p xs.storage r off).getD 0)

def Backing.query (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.Q (b.val*p.q.batch+h.val*p.q.head+m*p.q.row+e*p.q.dim)

def Backing.key (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.K (b.val*p.k.batch+h.val*p.k.head+e*p.k.dim+m*p.k.row)

def Backing.value (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.V (b.val*p.v.batch+h.val*p.v.head+m*p.v.row+e*p.v.dim)

def Backing.queryScale (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (block : Nat) : ℝ :=
  read xs p.QS ((b.val*p.heads+h.val)*((p.length+127)/128)+block)

def Backing.keyScale (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (block : Nat) : ℝ :=
  read xs p.KS ((b.val*p.heads+h.val)*((p.length+63)/64)+block)

/-- Mathematical input arrays in logical coordinates. Total extensions outside
finite windows retain the prior specification's exact alias convention. -/
structure Arrays (p : Parameters) where
  query : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  key : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  value : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  queryScale : Fin p.batches → Fin p.heads → Nat → ℝ
  keyScale : Fin p.batches → Fin p.heads → Nat → ℝ

def arraysOfBacking (p : Parameters) (xs : Backing p) : Arrays p where
  query := xs.query
  key := xs.key
  value := xs.value
  queryScale := xs.queryScale
  keyScale := xs.keyScale

/-- The logical arrays and complete original windows are tied by an equality.
This keeps aliases, padding and unused cells without treating them as new
independent mathematical inputs. -/
structure Inputs (p : Parameters) where
  arrays : Arrays p
  backing : Backing p
  connection : arrays = arraysOfBacking p backing

def Inputs.query (xs : Inputs p) := xs.arrays.query
def Inputs.key (xs : Inputs p) := xs.arrays.key
def Inputs.value (xs : Inputs p) := xs.arrays.value
def Inputs.queryScale (xs : Inputs p) := xs.arrays.queryScale
def Inputs.keyScale (xs : Inputs p) := xs.arrays.keyScale

def inputsEquiv (p : Parameters) (work : List Program) :
    (AttentionIO.realIO p work).Inputs ≃ Inputs p where
  toFun := fun values => ⟨arraysOfBacking p ⟨values⟩,⟨values⟩,rfl⟩
  invFun := fun xs => xs.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

def io (p : Parameters) (work : List Program) :
    LogicalProgramIO Real.R Real.R (Inputs p) (Outputs p) where
  core := AttentionIO.realIO p work
  inputs := inputsEquiv p work
  outputs := outputsEquiv p work

def finiteInputs (p : Parameters) (xs : Inputs p) : Prop :=
  AttentionIO.RealInputs p xs.backing.storage

end
end VeriTile.Bench.Repairs.AttentionForwardTriton.LogicalIO
