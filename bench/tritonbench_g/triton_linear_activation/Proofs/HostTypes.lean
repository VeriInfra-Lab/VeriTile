import bench.tritonbench_g.triton_linear_activation.Proofs.Materialize
import bench.tritonbench_g.triton_linear_activation.Proofs.InputCongruence
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

inductive Batch where
  | matrix (m : Nat)
  | batched (batch sequence : Nat)
  deriving DecidableEq

def Batch.rows : Batch → Nat
  | .matrix m => m
  | .batched b s => b*s

def Batch.shape (b : Batch) (last : Nat) : List Nat :=
  match b with
  | .matrix m => [m,last]
  | .batched b s => [b,s,last]

def canonicalStrides : List Nat → List Nat
  | [] => []
  | _::tail => (tail.map (max 1)).prod :: canonicalStrides tail

structure Tensor where
  objectId : Nat
  storage : Nat
  shape : List Nat
  strides : List Nat
  dtype : Mode
  device : Nat
  offset : Nat
  negative : Bool
  conjugate : Bool
  deriving DecidableEq

def tensor (objectId storage : Nat) (shape : List Nat) (dtype : Mode) (device : Nat) : Tensor :=
  ⟨objectId,storage,shape,canonicalStrides shape,dtype,device,0,false,false⟩

/-- Exactly the admitted caller domain: explicitly FP32, canonical rank-2 or
rank-3 X, W[N,K], optional bias[N], optional saved[M,N]. Mode denotes ambient
CUDA autocast; that original decorator branch always computes in FP16. -/
structure Request where
  batch : Batch
  n : Nat
  k : Nat
  device : Nat
  storage : Fin 4 → Nat
  object : Fin 4 → Nat
  hasBias : Bool
  hasSaved : Bool
  mode : Mode
  activation : Option String

def Request.shape (r : Request) : Shape := ⟨r.batch.rows,r.n,r.k⟩
def Request.count (r : Request) : Fin 4 → Nat :=
  ![r.batch.rows*r.k,r.n*r.k,if r.hasBias then r.n else 0,if r.hasSaved then r.batch.rows*r.n else 0]
def Request.argumentShape (r : Request) : Fin 4 → List Nat :=
  ![r.batch.shape r.k,[r.n,r.k],[r.n],[r.batch.rows,r.n]]
def Request.argument (r : Request) (i : Fin 4) : Tensor :=
  tensor (r.object i) (r.storage i) (r.argumentShape i) .fp32 r.device

def Request.Valid (r : Request) : Prop :=
  SourceBounds r.shape ∧ ∀ i j : Fin 4,i≠j → 0<r.count i → 0<r.count j → r.storage i≠r.storage j

/-- An ordinary fresh-allocation interface: arbitrary initial fresh contents,
exact caller frame, fresh object/storage identities. No copy, dot, result or
returned-value equation is an allocator premise. Slots 0..4 are output,
AMP X/W/bias and saved scratch; slot 5 is the optional rank-3 return view. -/
structure Allocation (r : Request) (caller : Set Nat) (before : Memory α) where
  storage : Fin 6 → Nat
  object : Fin 6 → Nat
  distinct : Function.Injective storage
  objectDistinct : Function.Injective object
  fresh : ∀ i,storage i∉caller
  objectFresh : ∀ i j,object i≠r.object j
  inputPresent : ∀ i,r.storage i∈caller
  initial : Memory α
  frame : ∀ p,p.1∈caller → initial p=before p

variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

def inputSlot (i : Fin 3) : Fin 4 := ⟨i.val,by omega⟩
def castSlot (i : Fin 3) : Fin 6 := ⟨i.val+1,by omega⟩

theorem castSlot_injective : Function.Injective castSlot := by
  intro i j h
  apply Fin.ext
  have := congrArg Fin.val h
  simp only [castSlot] at this
  omega

def Allocation.copies (allocation : Allocation r caller before) : CopyRegions 3 where
  src := fun i => r.storage (inputSlot i)
  dst := fun i => allocation.storage (castSlot i)
  count := fun i => r.count (inputSlot i)
  distinct := allocation.distinct.comp castSlot_injective
  separate := fun i j h => allocation.fresh (castSlot j) (h ▸ allocation.inputPresent (inputSlot i))

def Allocation.computeStorage (allocation : Allocation r caller before) (i : Fin 3) : Nat :=
  match r.mode with
  | .fp32 => r.storage (inputSlot i)
  | .autocastFP16 => allocation.storage (castSlot i)

def Allocation.buffers (allocation : Allocation r caller before) : Buffers :=
  ⟨allocation.computeStorage 0,allocation.computeStorage 1,
    if r.hasBias then some (allocation.computeStorage 2) else none,
    if r.hasSaved then some (match r.mode with
      | .fp32 => r.storage 3 | .autocastFP16 => allocation.storage 4) else none,
    allocation.storage 0⟩

def Allocation.prepared (allocation : Allocation r caller before) (a : Arithmetic α) : Memory α :=
  match r.mode with
  | .fp32 => allocation.initial
  | .autocastFP16 => allocation.copies.run a .fp32 .autocastFP16 allocation.initial

 theorem Allocation.fresh_input (allocation : Allocation r caller before) (i : Fin 6) (j : Fin 4) :
    allocation.storage i≠r.storage j := by
  intro h
  exact allocation.fresh i (h ▸ allocation.inputPresent j)

 theorem Allocation.prepared_frame (allocation : Allocation r caller before) (a : Arithmetic α)
    (p : Address) (hp : p.1∈caller) : allocation.prepared a p=before p := by
  cases hm : r.mode with
  | fp32 => simpa only [Allocation.prepared,hm] using allocation.frame p hp
  | autocastFP16 =>
    simp only [Allocation.prepared,hm]
    rw [allocation.copies.frame a .fp32 .autocastFP16 allocation.initial p]
    · exact allocation.frame p hp
    · intro i hi
      apply allocation.fresh (castSlot i)
      have eq : p.1=allocation.storage (castSlot i) := hi.1
      exact eq ▸ hp

 theorem Allocation.prepared_input (allocation : Allocation r caller before) (a : Arithmetic α)
    (i : Fin 3) (offset : Nat) (hi : offset<r.count (inputSlot i)) :
    allocation.prepared a (allocation.computeStorage i,offset)=
      (match r.mode with
      | .fp32 => before (r.storage (inputSlot i),offset)
      | .autocastFP16 => a.encode .autocastFP16 (a.decode .fp32 (before (r.storage (inputSlot i),offset)))) := by
  cases hm : r.mode with
  | fp32 =>
    simp only [Allocation.prepared,Allocation.computeStorage,hm]
    exact allocation.frame (r.storage (inputSlot i),offset) (allocation.inputPresent (inputSlot i))
  | autocastFP16 =>
    simp only [Allocation.prepared,Allocation.computeStorage,hm]
    calc
      _ = a.encode .autocastFP16 (a.decode .fp32 (allocation.initial (r.storage (inputSlot i),offset))) :=
        allocation.copies.output a .fp32 .autocastFP16 allocation.initial i offset hi
      _ = _ := by rw [allocation.frame (r.storage (inputSlot i),offset) (allocation.inputPresent (inputSlot i))]

#axiomsClean Allocation.prepared_frame
#axiomsClean Allocation.prepared_input
end
end VeriTile.Bench.Repairs.TritonLinearActivation
