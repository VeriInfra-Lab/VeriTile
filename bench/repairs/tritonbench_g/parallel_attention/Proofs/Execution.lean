import bench.repairs.tritonbench_g.parallel_attention.Proofs.Graph
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Memory

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

structure Bases where
  q : Nat
  k : Nat
  v : Nat
  do_ : Nat
  dz : Nat

def tensorCells (base M T D : Nat) : Set Nat :=
  {a | ∃ bh t e,bh<M ∧ t<T ∧ e<D ∧ a=base+dense T D bh t e}

def inputCells (s : Shape) (b : Bases) : Set Nat :=
  tensorCells b.q (s.B*s.H) s.T s.K ∪ tensorCells b.k (s.B*s.H) s.T s.K ∪
  tensorCells b.v (s.B*s.H) s.T s.V ∪ tensorCells b.do_ (s.B*s.H) s.T s.V ∪ tensorCells b.dz (s.B*s.H) s.T 1

def tensorRead (a : Ops α) (m : Memory α) (base M T D bh t e : Nat) : α :=
  if bh<M ∧ t<T ∧ e<D then m (base+dense T D bh t e) else a.zero

def currentInputs (a : Ops α) (s : Shape) (b : Bases) (m : Memory α) : TensorInputs α where
  q := tensorRead a m b.q (s.B*s.H) s.T s.K
  k := tensorRead a m b.k (s.B*s.H) s.T s.K
  v := tensorRead a m b.v (s.B*s.H) s.T s.V
  do_ := tensorRead a m b.do_ (s.B*s.H) s.T s.V
  dz := fun bh t => tensorRead a m b.dz (s.B*s.H) s.T 1 bh t 0

theorem tensorRead_congr (a : Ops α) (m m' : Memory α) (base M T D : Nat)
    (same : Set.EqOn m m' (tensorCells base M T D)) :
    tensorRead a m base M T D=tensorRead a m' base M T D := by
  funext bh t e
  unfold tensorRead
  split_ifs with h
  · exact same ⟨bh,t,e,h.1,h.2.1,h.2.2,rfl⟩
  · rfl

theorem currentInputs_congr (a : Ops α) (s : Shape) (b : Bases) (m m' : Memory α)
    (same : Set.EqOn m m' (inputCells s b)) :
    currentInputs a s b m=currentInputs a s b m' := by
  unfold currentInputs
  congr 1
  · apply tensorRead_congr
    intro i hi
    exact same (Or.inl (Or.inl (Or.inl (Or.inl hi))))
  · apply tensorRead_congr
    intro i hi
    exact same (Or.inl (Or.inl (Or.inl (Or.inr hi))))
  · apply tensorRead_congr
    intro i hi
    exact same (Or.inl (Or.inl (Or.inr hi)))
  · apply tensorRead_congr
    intro i hi
    exact same (Or.inl (Or.inr hi))
  · have h := tensorRead_congr a m m' b.dz (s.B*s.H) s.T 1 (by
      intro i hi
      exact same (Or.inr hi))
    rw [h]

inductive OutputKind where
  | forward | normalizer | query | key | value
  deriving DecidableEq,Repr

def OutputKind.width (kind : OutputKind) (s : Shape) :=
  match kind with | .forward | .value => s.V | .normalizer => 1 | .query | .key => s.K

def OutputKind.shards (kind : OutputKind) (s : Shape) :=
  match kind with | .forward | .normalizer | .value => 1 | .query | .key => blocks s.V s.BV

def OutputKind.count (kind : OutputKind) (s : Shape) :=
  kind.shards s*(s.B*s.H*s.T*kind.width s)

/-- Decode the unique owning launch program for every active scratch cell.
No value is computed from its old scratch contents. -/
def scratchValue (a : Ops α) (p : Plans) (s : Shape) (kind : OutputKind)
    (x : TensorInputs α) (i : Nat) : α :=
  let D := kind.width s
  let e := i%D
  let t := i/D%s.T
  let bh := i/D/s.T%(s.B*s.H)
  let shard := i/(s.B*s.H*s.T*D)
  match kind with
  | .forward => (forwardLane a p s x ⟨bh,t/128,0,e/s.BV⟩ (t%128) (e%s.BV)).1
  | .normalizer => (forwardLane a p s x ⟨bh,t/128,0,0⟩ (t%128) 0).2
  | .query => queryLane a p s x ⟨bh,t/64,0,shard⟩ (t%64) e
  | .key => (keyValueLane a p s x ⟨bh,t/64,0,shard⟩ (t%64) e 0).1
  | .value => (keyValueLane a p s x ⟨bh,t/64,0,e/s.BV⟩ (t%64) 0 (e%s.BV)).2

def scratchCurrent (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (m : Memory α) (i : Nat) : α := scratchValue a p s kind (currentInputs a s b m) i

theorem scratch_local (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind) :
    ReadsOnly (inputCells s b) (scratchCurrent a p s b kind) := by
  intro m m' same i
  unfold scratchCurrent
  rw [currentInputs_congr a s b m m' same]

/-- Any serialization of active stores gives the same memory result. This
models arbitrary scheduling, including the required empty-grid no-op. -/
def launch (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (base : Nat) (schedule : List Nat) (m : Memory α) :=
  runStores (fun i => base+i) (scratchCurrent a p s b kind) schedule m

theorem launch_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (base : Nat) (schedule : List Nat) (covers : schedule.Perm (List.range (kind.count s)))
    (fresh : ∀ i, i<kind.count s → base+i∉inputCells s b) (m : Memory α)
    (i : Nat) (hi : i<kind.count s) :
    launch a p s b kind base schedule m (base+i)=scratchValue a p s kind (currentInputs a s b m) i := by
  apply actual_store_value (inputCells s b) (fun i => base+i) (scratchCurrent a p s b kind)
    (scratch_local a p s b kind) schedule m
  · intro j hj
    exact fresh j (List.mem_range.mp (covers.mem_iff.mp hj))
  · exact covers.mem_iff.mpr (List.mem_range.mpr hi)
  · intro i j hij
    dsimp only at hij
    omega

theorem launch_frame (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (base : Nat) (schedule : List Nat) (covers : schedule.Perm (List.range (kind.count s)))
    (m : Memory α) (address : Nat) (outside : address<base ∨ base+kind.count s≤address) :
    launch a p s b kind base schedule m address=m address := by
  apply runStores_frame
  intro i hi
  have hb := List.mem_range.mp (covers.mem_iff.mp hi)
  omega

/-- Host sum(0) reads the actual completed scratch memory, then performs the
source output cast. Its tree is finite syntax supplied by the Torch lowering. -/
def hostValue (a : Ops α) (tree : SumPlan) (shards base cells : Nat) (m : Memory α) (i : Nat) : α :=
  a.cast (tree.eval a (fun shard => if shard<shards ∧ i<cells then m (base+shard*cells+i) else a.zero))

def hostRun (a : Ops α) (tree : SumPlan) (shards scratch output cells : Nat) (m : Memory α) :=
  runStores (fun i => output+i) (fun mem i => hostValue a tree shards scratch cells mem i) (List.range cells) m

def scratchCells (shards base cells : Nat) : Set Nat :=
  {address | ∃ shard i,shard<shards ∧ i<cells ∧ address=base+shard*cells+i}

theorem host_local (a : Ops α) (tree : SumPlan) (shards scratch cells : Nat) :
    ReadsOnly (scratchCells shards scratch cells)
      (fun mem i => hostValue a tree shards scratch cells mem i) := by
  intro m m' same i
  have eq : (fun shard => if shard<shards ∧ i<cells then m (scratch+shard*cells+i) else a.zero)=
      (fun shard => if shard<shards ∧ i<cells then m' (scratch+shard*cells+i) else a.zero) := by
    funext shard
    split_ifs with h
    · exact same ⟨shard,i,h.1,h.2,rfl⟩
    · rfl
  simpa only [hostValue] using congrArg (fun x : Nat → α => a.cast (tree.eval a x)) eq

theorem host_output (a : Ops α) (tree : SumPlan) (shards scratch output cells : Nat)
    (m : Memory α) (fresh : ∀ i, i<cells → output+i∉scratchCells shards scratch cells)
    (i : Nat) (hi : i<cells) :
    hostRun a tree shards scratch output cells m (output+i)=hostValue a tree shards scratch cells m i := by
  apply actual_store_value (scratchCells shards scratch cells) (fun i => output+i)
    (fun mem i => hostValue a tree shards scratch cells mem i)
    (host_local a tree shards scratch cells) (List.range cells) m
  · intro j hj
    exact fresh j (List.mem_range.mp hj)
  · exact List.mem_range.mpr hi
  · intro j k eq
    dsimp only at eq
    omega

theorem host_frame (a : Ops α) (tree : SumPlan) (shards scratch output cells : Nat)
    (m : Memory α) (address : Nat) (outside : address<output ∨ output+cells≤address) :
    hostRun a tree shards scratch output cells m address=m address := by
  apply runStores_frame
  intro i hi
  have hb := List.mem_range.mp hi
  omega

/-- The exact same current-memory execution is preserved by scalar backend
agreement. This does not identify floating point with real arithmetic. -/
theorem launch_backend (a bops : Ops α) (same : ScalarAgreement a bops) (p : Plans)
    (s : Shape) (b : Bases) (kind : OutputKind) (base : Nat) (schedule : List Nat) (m : Memory α) :
    launch a p s b kind base schedule m=launch bops p s b kind base schedule m := by
  rw [same.eq]

#axiomsClean currentInputs_congr
#axiomsClean scratch_local
#axiomsClean launch_output
#axiomsClean launch_frame
#axiomsClean launch_backend
#axiomsClean host_output
#axiomsClean host_frame
end
end VeriTile.Bench.Repairs.ParallelAttention
