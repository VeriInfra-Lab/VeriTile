import bench.tritonbench_g.parallel_retention_attention.Proofs.Runtime
import bench.tritonbench_g.parallel_retention_attention.Proofs.Domain

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- The public family is FP32 Q,K:[B,H,T,DK] and V,G:[B,H,T,DV], on one
CUDA device. The existing contiguous decorator materializes their logical
values; nonzero storage offsets are included in each base. DK,DV are positive;
B,H,T may be zero. This is the source's exact tile choice and index domain. -/
structure ShapeContract (s : Shape) : Prop where
  keyTile : s.BK=featureWidth s.DK
  valueTile : s.BV=featureWidth s.DV
  indices : IndexDomain s.B s.H s.T s.DK s.DV s.BK s.BV

/-- A four-dimensional public tensor coordinate. -/
structure LogicalIndex (s : Shape) (kind : OutputKind) where
  batch : Fin s.B
  head : Fin s.H
  time : Fin s.T
  feature : Fin (kind.width s)

def LogicalIndex.bh (i : LogicalIndex s kind) : Nat := i.batch.val*s.H+i.head.val

theorem LogicalIndex.bh_bound (i : LogicalIndex s kind) : i.bh<s.B*s.H := by
  have hb := i.batch.isLt
  have hh := i.head.isLt
  unfold LogicalIndex.bh
  nlinarith

theorem LogicalIndex.head_eq (i : LogicalIndex s kind) : i.bh%s.H=i.head.val := by
  simp [LogicalIndex.bh,Nat.add_mod,Nat.mod_eq_of_lt i.head.isLt]

/-- Caller-visible logical input tensors, independent of all tile dimensions,
reduction trees, addresses, scratch arrays and loop states. B,H are flattened
only inside this storage-independent representation. -/
structure LogicalInputs (α : Type) (s : Shape) where
  q : Fin (s.B*s.H) → Fin s.T → Fin s.DK → α
  k : Fin (s.B*s.H) → Fin s.T → Fin s.DK → α
  v : Fin (s.B*s.H) → Fin s.T → Fin s.DV → α
  g : Fin (s.B*s.H) → Fin s.T → Fin s.DV → α

def paddedTensor (zero : α) (M T D : Nat) (x : Fin M → Fin T → Fin D → α)
    (bh t e : Nat) : α :=
  if h : bh<M ∧ t<T ∧ e<D then x ⟨bh,h.1⟩ ⟨t,h.2.1⟩ ⟨e,h.2.2⟩ else zero

def LogicalInputs.padded (a : Ops α) (x : LogicalInputs α s) : Inputs α where
  q := paddedTensor a.zero (s.B*s.H) s.T s.DK x.q
  k := paddedTensor a.zero (s.B*s.H) s.T s.DK x.k
  v := paddedTensor a.zero (s.B*s.H) s.T s.DV x.v
  g := paddedTensor a.zero (s.B*s.H) s.T s.DV x.g

/-- Ordinary input-only contiguous-copy/read contract. In particular, singleton
strides need not equal canonical strides: only active logical coordinates are
bound. There is no launch, result or final-memory premise. -/
structure ForwardMaterialized (s : Shape) (b : Bases) (m : Memory α) (x : LogicalInputs α s) : Prop where
  query : ∀ bh t e,m (b.q+dense s.T s.DK bh.val t.val e.val)=x.q bh t e
  key : ∀ bh t e,m (b.k+dense s.T s.DK bh.val t.val e.val)=x.k bh t e
  value : ∀ bh t e,m (b.v+dense s.T s.DV bh.val t.val e.val)=x.v bh t e
/-- Backward additionally binds its actual cotangent argument. -/
structure Materialized (s : Shape) (b : Bases) (m : Memory α) (x : LogicalInputs α s)
    : Prop extends ForwardMaterialized s b m x where
  cotangent : ∀ bh t e,m (b.g+dense s.T s.DV bh.val t.val e.val)=x.g bh t e

theorem materialized_read (a : Ops α) (s : Shape) (b : Bases) (m : Memory α)
    (x : LogicalInputs α s) (bound : Materialized s b m x) : currentInputs a s b m=x.padded a := by
  unfold currentInputs LogicalInputs.padded
  congr 1
  all_goals
    funext bh t e
    unfold tensorRead paddedTensor
    split_ifs with h
    · first | exact bound.query ⟨bh,h.1⟩ ⟨t,h.2.1⟩ ⟨e,h.2.2⟩
            | exact bound.key ⟨bh,h.1⟩ ⟨t,h.2.1⟩ ⟨e,h.2.2⟩
            | exact bound.value ⟨bh,h.1⟩ ⟨t,h.2.1⟩ ⟨e,h.2.2⟩
            | exact bound.cotangent ⟨bh,h.1⟩ ⟨t,h.2.1⟩ ⟨e,h.2.2⟩
    · rfl

/-- Forward has no cotangent argument or cotangent read footprint. -/
def forwardInputCells (s : Shape) (b : Bases) : Set Nat :=
  tensorCells b.q (s.B*s.H) s.T s.DK ∪ tensorCells b.k (s.B*s.H) s.T s.DK ∪
  tensorCells b.v (s.B*s.H) s.T s.DV

theorem materialized_tensor (a : Ops α) (m : Memory α) (base M T D : Nat)
    (x : Fin M → Fin T → Fin D → α)
    (bound : ∀ bh t e,m (base+dense T D bh.val t.val e.val)=x bh t e) :
    tensorRead a m base M T D=paddedTensor a.zero M T D x := by
  funext bh t e
  unfold tensorRead paddedTensor
  split_ifs with h
  · exact bound ⟨bh,h.1⟩ ⟨t,h.2.1⟩ ⟨e,h.2.2⟩
  · rfl

theorem materialized_forward_reads (a : Ops α) (s : Shape) (b : Bases) (m : Memory α)
    (x : LogicalInputs α s) (bound : ForwardMaterialized s b m x) :
    (currentInputs a s b m).q=(x.padded a).q ∧
    (currentInputs a s b m).k=(x.padded a).k ∧
    (currentInputs a s b m).v=(x.padded a).v :=
  ⟨materialized_tensor a m _ _ _ _ x.q bound.query,
   materialized_tensor a m _ _ _ _ x.k bound.key,
   materialized_tensor a m _ _ _ _ x.v bound.value⟩

theorem forward_scratch_inputs (a : Ops α) (p : Plans) (s : Shape) (x y : Inputs α)
    (hq : x.q=y.q) (hk : x.k=y.k) (hv : x.v=y.v) (i : Nat) :
    scratchValue a p s .forward x i=scratchValue a p s .forward y i := by
  have stepEq : forwardStep a p s x=forwardStep a p s y := by
    funext tile r v diagonal j carry
    simp only [forwardStep,hq,hk,hv]
  simp only [scratchValue,forwardLane,stepEq]

theorem forward_word_inputs (a : Ops α) (p : Plans) (s : Shape) (tree : SumPlan)
    (x y : Inputs α) (hq : x.q=y.q) (hk : x.k=y.k) (hv : x.v=y.v) (i : Nat) :
    wordReference a p s .forward tree x i=wordReference a p s .forward tree y i := by
  simp only [wordReference,forward_scratch_inputs a p s x y hq hk hv]

theorem forward_scratch_local (a : Ops α) (p : Plans) (s : Shape) (b : Bases) :
    ReadsOnly (forwardInputCells s b) (scratchCurrent a p s b .forward) := by
  intro m m' same i
  apply forward_scratch_inputs
  · apply tensorRead_congr
    intro address ha
    exact same (Or.inl (Or.inl ha))
  · apply tensorRead_congr
    intro address ha
    exact same (Or.inl (Or.inr ha))
  · apply tensorRead_congr
    intro address ha
    exact same (Or.inr ha)

structure ForwardLayout (s : Shape) (b : Bases) where
  scratch : Nat
  output : Nat
  scratchFresh : ∀ i,i<OutputKind.forward.count s → scratch+i∉forwardInputCells s b
  outputFresh : ∀ i,i<OutputKind.forward.cells s →
    output+i∉scratchCells (OutputKind.forward.shards s) scratch (OutputKind.forward.cells s)

structure BackwardLayout (s : Shape) (b : Bases) where
  scratch : OutputKind → Nat
  output : OutputKind → Nat
  scratchSeparate : RegionDisjoint s true scratch
  outputSeparate : RegionDisjoint s false output
  scratchFresh : ∀ cell∈allCells s true,cellsAddress scratch cell∉inputCells s b
  outputFresh : ∀ cell∈allCells s false,cellsAddress output cell∉backwardScratchCells s scratch

/-- Metadata of the tensors saved after the source's existing contiguous
materialization. No arbitrary-stride kernel API is introduced. -/
structure SavedInputs (s : Shape) (b : Bases) where
  query : TensorHandle
  key : TensorHandle
  value : TensorHandle
  queryShape : query.shape=[s.B,s.H,s.T,s.DK]
  keyShape : key.shape=[s.B,s.H,s.T,s.DK]
  valueShape : value.shape=[s.B,s.H,s.T,s.DV]
  fp32 : query.dtypeFP32=true ∧ key.dtypeFP32=true ∧ value.dtypeFP32=true
  devices : key.device=query.device ∧ value.device=query.device
  bases : query.base=b.q ∧ key.base=b.k ∧ value.base=b.v

def readReturned (s : Shape) (kind : OutputKind) (handle : TensorHandle)
    (m : Memory α) (i : LogicalIndex s kind) : α :=
  m (handle.base+dense s.T (kind.width s) i.bh i.time i.feature)

#axiomsClean forward_scratch_local
#axiomsClean materialized_forward_reads
#axiomsClean forward_word_inputs
#axiomsClean materialized_read
#axiomsClean LogicalIndex.bh_bound
#axiomsClean LogicalIndex.head_eq
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
