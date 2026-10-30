import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Native
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Real
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

abbrev Row (p : Parameters) := Fin (ceil p.length 16)
abbrev Head (p : Parameters) (batches : Nat) := Fin (batches*p.heads)

/-- Finite actual-read windows. A CSR occurrence is retained separately even
when another occurrence names the same key. The visit capacity is explicit;
only the signed endpoint interval enables its lanes. -/
@[reducible] def inputs (p : Parameters) (batches visits : Nat) : List IOChannel :=
  [⟨.int,[ceil p.length 16,batches*p.heads,2],"rows"⟩,
   ⟨.int,[ceil p.length 16,batches*p.heads,visits],"cols"⟩,
   ⟨.real,[ceil p.length 16,batches*p.heads,visits,p.blocks,16,16],"Q"⟩,
   ⟨.real,[ceil p.length 16,batches*p.heads,visits,p.blocks,16,16],"K"⟩,
   ⟨.real,[ceil p.length 16,batches*p.heads,p.blocks,visits,16,16],"V"⟩]

@[reducible] def outputs (p : Parameters) (batches : Nat) : List IOChannel :=
  [⟨.real,[ceil p.length 16,batches*p.heads,p.blocks,16,16],"out"⟩]

abbrev Inputs (α : Type) (p : Parameters) (batches visits : Nat) :=
  ChannelValues α (inputs p batches visits)
abbrev Outputs (α : Type) (p : Parameters) (batches : Nat) :=
  ChannelValues α (outputs p batches)

def first {α : Type} {p : Parameters} {batches visits : Nat}
    (xs : Inputs α p batches visits) (r : Row p) (h : Head p batches) : Int :=
  xs 0 (r,h,0,PUnit.unit)
def last {α : Type} {p : Parameters} {batches visits : Nat}
    (xs : Inputs α p batches visits) (r : Row p) (h : Head p batches) : Int :=
  xs 0 (r,h,1,PUnit.unit)
def count {α : Type} {p : Parameters} {batches visits : Nat}
    (xs : Inputs α p batches visits) (r : Row p) (h : Head p batches) : Nat :=
  (last xs r h-first xs r h).toNat
def column {α : Type} {p : Parameters} {batches visits : Nat}
    (xs : Inputs α p batches visits) (r : Row p) (h : Head p batches)
    (v : Fin visits) : Int := xs 1 (r,h,v,PUnit.unit)

def head (p : Parameters) (h : Nat) : Nat := h%p.heads
def batch (p : Parameters) (h : Nat) : Nat := h/p.heads
def kv (p : Parameters) (h : Nat) : Nat := head p h/(p.heads/p.kvHeads)
def layout (p : Parameters) (h : Nat) : Nat := head p h%p.layouts

def inputOffset {α : Type} (p : Parameters) (batches visits : Nat)
    (port : Fin (inputs p batches visits).length) (xs : Inputs α p batches visits) :
    TileIndex (inputs p batches visits)[port].shape → Nat := match port with
  | ⟨0,_⟩ => fun j => layout p j.2.1.val*p.rowStride+j.1.val+j.2.2.1.val
  | ⟨1,_⟩ => fun j => ((layout p j.2.1.val*p.colStride : Nat)+
      first xs j.1 j.2.1+j.2.2.1.val : Int).toNat
  | ⟨2,_⟩ => fun j => batch p j.2.1.val*p.qb+head p j.2.1.val*p.qh+
      (j.1.val*16+j.2.2.2.2.1.val)*p.qm+j.2.2.2.2.2.1.val*p.qd+
      j.2.2.2.1.val*16*p.qd
  | ⟨3,_⟩ => fun j => batch p j.2.1.val*p.kb+kv p j.2.1.val*p.kh+
      j.2.2.2.2.2.1.val*p.kn+j.2.2.2.2.1.val*p.kd+
      (column xs j.1 j.2.1 j.2.2.1*16*p.kn).toNat+j.2.2.2.1.val*16*p.kd
  | ⟨4,_⟩ => fun j => batch p j.2.1.val*p.vb+kv p j.2.1.val*p.vh+
      j.2.2.2.2.1.val*p.vn+(j.2.2.1.val*16+j.2.2.2.2.2.1.val)*p.vd+
      (column xs j.1 j.2.1 j.2.2.2.1*16*p.vn).toNat

def inputActive {α : Type} (p : Parameters) (batches visits : Nat)
    (port : Fin (inputs p batches visits).length) (xs : Inputs α p batches visits) :
    TileIndex (inputs p batches visits)[port].shape → Prop := match port with
  | ⟨0,_⟩ => fun _ => 0<p.blocks
  | ⟨1,_⟩ => fun j => 0<p.blocks ∧ j.2.2.1.val<count xs j.1 j.2.1
  | ⟨2,_⟩ => fun j => j.2.2.1.val<count xs j.1 j.2.1 ∧
      (p.evenM=Bool.true ∨ j.1.val*16+j.2.2.2.2.1.val<p.length)
  | ⟨3,_⟩ => fun j => j.2.2.1.val<count xs j.1 j.2.1 ∧
      (p.evenN=Bool.true ∨ (j.2.2.2.2.2.1.val : Int)+column xs j.1 j.2.1 j.2.2.1*16<p.length)
  | ⟨4,_⟩ => fun j => j.2.2.2.1.val<count xs j.1 j.2.1 ∧
      (p.evenN=Bool.true ∨ (j.2.2.2.2.1.val : Int)+column xs j.1 j.2.1 j.2.2.2.1*16<p.length)

def outputOffset (p : Parameters) (batches : Nat)
    (j : TileIndex [ceil p.length 16,batches*p.heads,p.blocks,16,16]) : Nat :=
  batch p j.2.1.val*p.ob+head p j.2.1.val*p.oh+
    (j.1.val*16+j.2.2.2.1.val)*p.om+j.2.2.1.val*16+j.2.2.2.2.1.val

/-- Ordinary scalar geometry retained from RawABI, without a Tensor premise. -/
def Geometry (p : Parameters) : Prop :=
  0<p.heads ∧ 16≤p.om ∧ p.ob=p.heads*p.length*(16*p.blocks) ∧
    p.oh=p.length*(16*p.blocks) ∧ p.om=16*p.blocks

/-- Metadata domain only: actual signed INT32 reads, the original nonnegative
first/column conditions, and enough finite lanes for all CSR occurrences.
Reversed endpoints still execute zero visits. Backing is supplied separately
by the shared InBounds judgment, never hidden in this predicate. -/
def Admissible {α : Type} (p : Parameters) (batches visits : Nat)
    (xs : Inputs α p batches visits) : Prop :=
  ∀ r h,0<p.blocks →
    Int32Bits.inRange (first xs r h) ∧ Int32Bits.inRange (last xs r h) ∧
    0≤first xs r h ∧ count xs r h≤visits ∧
    ∀ v : Fin visits,v.val<count xs r h →
      Int32Bits.inRange (column xs r h v) ∧ 0≤column xs r h v

def invocation {α : Type} (a : Arithmetic α) (native : Bool)
    (p : Parameters) (id : PhysicalKernel.Program) : Invocation (Replicas α) where
  kernel := if native then .mk (source p).inputs (source p).outputs (selectedSource p) else source p
  backend := a.model
  arguments := [⟨"softmax_scale",.real,[],fun _=>broadcast (a.argument p.scale)⟩]
  clearRegisters := Bool.true
  pids := id.pids
  numPids := some (fun _=>0)

/-- The complete original three-axis grid; runtime FP32 scale and selected
native expression-site lowering remain explicit in each actual invocation. -/
def io {α β : Type} (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) : ProgramIO α (Replicas β) where
  inputs := inputs p batches visits
  outputs := outputs p batches
  scratch := []
  buffers := ["rows","cols","Q","K","V","out"]
  read := inputOffset p batches visits
  inputMask := inputActive p batches visits
  write := fun | ⟨0,_⟩ => fun _ => outputOffset p batches
  outputMask := fun | ⟨0,_⟩ => fun _ j=>j.1.val*16+j.2.2.2.1.val<p.length
  scratchWrite := fun i=>i.elim0
  scratchMask := fun i=>i.elim0
  schedule := fun _ => (PhysicalKernel.programs p batches).map (invocation a native p)
  encode := fun x=>broadcast (encode x)
  observe := fun | ⟨0,_⟩ => fun j x=>observe (x (owner j.2.2.2.2.1.val))

def realIO (p : Parameters) (batches visits : Nat) : ProgramIO ℝ (Replicas R) :=
  io mathArithmetic Bool.false some (WithBot.unbotD 0) p batches visits

def nativeIO (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (batches visits : Nat) :
    ProgramIO Word (Replicas Word) := io ops.arithmetic Bool.true id id p batches visits

end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

/-- Read a finite logical channel by its declared address. Repeated aliases
choose one representative; input binding proves that every enabled alias has
the same value. Addresses outside the declared reads use the explicit default.
This is a pure array view, with no source execution or mutable state. -/
def lookup {ι α : Type} (active : ι→Prop) (offset : ι→Nat) (value : ι→α)
    (fallback : α) (address : Nat) : α := by
  classical
  exact if found : ∃ j,active j ∧ offset j=address then value (Classical.choose found) else fallback

def rows {α : Type} (p : Parameters) (batches visits : Nat)
    (xs : Inputs α p batches visits) (address : Int) : Int :=
  lookup (inputActive p batches visits 0 xs) (inputOffset p batches visits 0 xs)
    (xs 0) 0 address.toNat
def columns {α : Type} (p : Parameters) (batches visits : Nat)
    (xs : Inputs α p batches visits) (address : Int) : Int :=
  lookup (inputActive p batches visits 1 xs) (inputOffset p batches visits 1 xs)
    (xs 1) 0 address.toNat

def memory {α : Type} (zero : α) (p : Parameters) (batches visits : Nat)
    (xs : Inputs α p batches visits) : BlockSparseAttn.Memory α
  | .q,address => lookup (inputActive p batches visits 2 xs) (inputOffset p batches visits 2 xs)
      (xs 2) zero address
  | .k,address => lookup (inputActive p batches visits 3 xs) (inputOffset p batches visits 3 xs)
      (xs 3) zero address
  | .v,address => lookup (inputActive p batches visits 4 xs) (inputOffset p batches visits 4 xs)
      (xs 4) zero address
  | _,_ => zero

def targetLayout {α : Type} (p : Parameters) (batches visits : Nat)
    (xs : Inputs α p batches visits) : BlockSparseAttn.Layout :=
  layoutOf p batches (rows p batches visits xs) (columns p batches visits xs)

def coordinate (p : Parameters) (batches : Nat)
    (j : TileIndex [ceil p.length 16,batches*p.heads,p.blocks,16,16]) : Work :=
  ⟨batch p j.2.1.val,head p j.2.1.val,j.1.val*16+j.2.2.2.1.val,
    j.2.2.1.val*16+j.2.2.2.2.1.val⟩

end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

/-- Each CSR occurrence, feature block, lane and masked tail remains an
independent value. These arrays do not identify repeated memory aliases. -/
structure LogicalInputs (p : Parameters) (batches visits : Nat) where
  rows : Row p → Head p batches → Fin 2 → Int
  columns : Row p → Head p batches → Fin visits → Int
  query : Row p → Head p batches → Fin visits → Fin p.blocks → Fin 16 → Fin 16 → ℝ
  key : Row p → Head p batches → Fin visits → Fin p.blocks → Fin 16 → Fin 16 → ℝ
  value : Row p → Head p batches → Fin p.blocks → Fin visits → Fin 16 → Fin 16 → ℝ

def logicalInputsEquiv (p : Parameters) (batches visits : Nat) :
    Inputs ℝ p batches visits ≃ LogicalInputs p batches visits where
  toFun := fun xs => ⟨fun r h e => xs 0 (r,h,e,()), fun r h v => xs 1 (r,h,v,()),
    fun r h v b i j => xs 2 (r,h,v,b,i,j,()),
    fun r h v b i j => xs 3 (r,h,v,b,i,j,()),
    fun r h b v i j => xs 4 (r,h,b,v,i,j,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.rows j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => xs.columns j.1 j.2.1 j.2.2.1)
      (Fin.cases (fun j => xs.query j.1 j.2.1 j.2.2.1 j.2.2.2.1 j.2.2.2.2.1 j.2.2.2.2.2.1)
        (Fin.cases (fun j => xs.key j.1 j.2.1 j.2.2.1 j.2.2.2.1 j.2.2.2.2.1 j.2.2.2.2.2.1)
          (Fin.cases (fun j => xs.value j.1 j.2.1 j.2.2.1 j.2.2.2.1 j.2.2.2.2.1 j.2.2.2.2.2.1)
            (fun i => i.elim0)))))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    · rcases j with ⟨r,h,e,⟨⟩⟩; rfl
    · rcases j with ⟨r,h,v,⟨⟩⟩; rfl
    · rcases j with ⟨r,h,v,b,i,j,⟨⟩⟩; rfl
    · rcases j with ⟨r,h,v,b,i,j,⟨⟩⟩; rfl
    · rcases j with ⟨r,h,b,v,i,j,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

abbrev Coordinate (p : Parameters) (batches : Nat) :=
  Head p batches × Fin (ceil p.length 16*16) × Fin (p.blocks*16)

def logicalIndex (p : Parameters) (batches : Nat) :
    TileIndex [ceil p.length 16,batches*p.heads,p.blocks,16,16] ≃ Coordinate p batches where
  toFun := fun j => (j.2.1,finProdFinEquiv (j.1,j.2.2.2.1),finProdFinEquiv (j.2.2.1,j.2.2.2.2.1))
  invFun := fun c => ((finProdFinEquiv.symm c.2.1).1,c.1,(finProdFinEquiv.symm c.2.2).1,
    (finProdFinEquiv.symm c.2.1).2,(finProdFinEquiv.symm c.2.2).2,())
  left_inv := by
    intro j
    rcases j with ⟨r,h,b,i,j,⟨⟩⟩
    change ((finProdFinEquiv.symm (finProdFinEquiv (r,i))).1,h,
      (finProdFinEquiv.symm (finProdFinEquiv (b,j))).1,
      (finProdFinEquiv.symm (finProdFinEquiv (r,i))).2,
      (finProdFinEquiv.symm (finProdFinEquiv (b,j))).2,PUnit.unit) = _
    rw [Equiv.symm_apply_apply,Equiv.symm_apply_apply]
  right_inv := by
    intro c
    change (c.1,finProdFinEquiv (finProdFinEquiv.symm c.2.1),
      finProdFinEquiv (finProdFinEquiv.symm c.2.2)) = c
    rw [Equiv.apply_symm_apply,Equiv.apply_symm_apply]

def logicalOutputsEquiv (p : Parameters) (batches : Nat) :
    Outputs ℝ p batches ≃ (Coordinate p batches → ℝ) where
  toFun := fun ys c => ys 0 ((logicalIndex p batches).symm c)
  invFun := fun f => Fin.cases (fun j => f (logicalIndex p batches j)) (fun i => i.elim0)
  left_inv := by
    intro ys
    funext o j
    fin_cases o
    exact congrArg (ys 0) ((logicalIndex p batches).symm_apply_apply j)
  right_inv := by intro f; funext c; exact congrArg f ((logicalIndex p batches).apply_symm_apply c)

def LogicalInputs.layout {p : Parameters} {batches visits : Nat}
    (xs : LogicalInputs p batches visits) : BlockSparseAttn.Layout :=
  targetLayout p batches visits ((logicalInputsEquiv p batches visits).symm xs)
def LogicalInputs.tensors {p : Parameters} {batches visits : Nat}
    (xs : LogicalInputs p batches visits) : BlockSparseAttn.Inputs :=
  BlockSparseAttn.Real.inputs xs.layout (memory 0 p batches visits ((logicalInputsEquiv p batches visits).symm xs))
def logicalDomain (p : Parameters) (batches visits : Nat) (xs : LogicalInputs p batches visits) : Prop :=
  Admissible p batches visits ((logicalInputsEquiv p batches visits).symm xs)
def Coordinate.work {p : Parameters} {batches : Nat} (c : Coordinate p batches) : Work :=
  coordinate p batches ((logicalIndex p batches).symm c)

def logicalIO (p : Parameters) (batches visits : Nat) :
    LogicalProgramIO ℝ (Replicas R) (LogicalInputs p batches visits) (Coordinate p batches → ℝ) where
  core := realIO p batches visits
  inputs := logicalInputsEquiv p batches visits
  outputs := logicalOutputsEquiv p batches

end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
