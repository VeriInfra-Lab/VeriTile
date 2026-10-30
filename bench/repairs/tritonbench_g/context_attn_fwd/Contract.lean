import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FormedBounds
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Checked
open VeriTile.Triton
set_option maxRecDepth 24000
set_option synthInstance.maxSize 512

/-- Direct observations of each actual Tensor and its full backing interval.
The call's shape/stride descriptors are obtained from these same Tensor API
queries; they contain no payload values or kernel-result claims. -/
structure View where
  metadata : VeriTile.Torch.TensorMetadata
  strides : List Nat
  (storageBase storageBytes storageOffset : Nat)
  conjugate : Bool
  deriving DecidableEq

structure Views where
  (q k v o start sequence prompt : View)

def all (v : Views) : List View := [v.q,v.k,v.v,v.o,v.start,v.sequence,v.prompt]
def reads (v : Views) : List View := [v.q,v.k,v.v,v.start,v.sequence,v.prompt]
def separate (a b : View) : Prop :=
  a.storageBytes=0 ∨ b.storageBytes=0 ∨ a.storageBase+a.storageBytes ≤ b.storageBase ∨
    b.storageBase+b.storageBytes ≤ a.storageBase

def cacheDense (v : Geometry.Original.View4) : Prop :=
  v.batches*v.heads*v.rows*v.dims=0 ∨
    v.dimStride=1 ∧ (1<v.rows → v.rowStride=v.dims) ∧
    (1<v.heads → v.headStride=v.rows*v.dims) ∧
    (1<v.batches → v.heads*v.rows*v.dims ≤ v.batchStride)

/-- Same-call descriptor observation: the source scalar stride arguments are
exactly the strides queried from each supplied Tensor, including cache batch gaps. -/
def strideChecks (c : Host.Call) (v : Views) : Prop :=
  v.q.strides=[c.args.q.rowStride,c.args.q.headStride,c.args.q.dimStride] ∧
  v.k.strides=[c.args.k.batchStride,c.args.k.headStride,c.args.k.rowStride,c.args.k.dimStride] ∧
  v.v.strides=[c.args.v.batchStride,c.args.v.headStride,c.args.v.rowStride,c.args.v.dimStride] ∧
  v.o.strides=[c.args.o.rowStride,c.args.o.headStride,c.args.o.dimStride]

instance (c : Host.Call) (v : Views) : Decidable (strideChecks c v) := by unfold strideChecks; infer_instance

def tensorChecks (c : Host.Call) (v : Views) : Prop :=
  v.q.metadata.shape=[c.args.q.rows,c.args.q.heads,c.args.q.dims] ∧
  v.k.metadata.shape=[c.args.k.batches,c.args.k.heads,c.args.k.rows,c.args.k.dims] ∧
  v.v.metadata.shape=[c.args.v.batches,c.args.v.heads,c.args.v.rows,c.args.v.dims] ∧
  v.o.metadata.shape=[c.args.o.rows,c.args.o.heads,c.args.o.dims] ∧
  (∀ x∈[v.start,v.sequence,v.prompt],x.metadata.shape=[c.batches] ∧
    x.metadata.dtype=.other "int32" ∧ x.metadata.isContiguous=Bool.true) ∧
  (∀ x∈[v.q,v.k,v.v,v.o],x.metadata.dtype=.floating .fp32) ∧
  v.q.metadata.device.kind="cuda" ∧
  (∀ x∈all v,x.metadata.layout=.strided ∧ x.metadata.device=v.q.metadata.device ∧
    x.metadata.isNeg=Bool.false ∧ x.conjugate=Bool.false ∧
    x.storageBytes%4=0 ∧ (x.metadata.numel=0 ∨ x.metadata.dataPtr.toNat%4=0)) ∧
  v.q.metadata.isContiguous=Bool.true ∧ v.o.metadata.isContiguous=Bool.true ∧
  cacheDense c.args.k ∧ cacheDense c.args.v ∧
  (c.batches=0 ∨ ∀ x∈reads v,separate v.o x) ∧ strideChecks c v

theorem source_strides (c : Host.Call) (v : Views) (h : tensorChecks c v) : strideChecks c v := by
  rcases h with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,strides⟩
  exact strides

def observedOffset3 (v : View) (row head dim : Nat) : Nat :=
  row*v.strides[0]?.getD 0+head*v.strides[1]?.getD 0+dim*v.strides[2]?.getD 0

def observedOffset4 (v : View) (batch head row dim : Nat) : Nat :=
  batch*v.strides[0]?.getD 0+head*v.strides[1]?.getD 0+
    row*v.strides[2]?.getD 0+dim*v.strides[3]?.getD 0

/-- Every source payload offset uses the same coordinate map as the supplied
view. In particular, K/V batch gaps cannot differ between checked view and source. -/
theorem source_view_offsets (c : Host.Call) (v : Views) (h : tensorChecks c v)
    (batch row head dim : Nat) :
    observedOffset3 v.q row head dim=row*c.args.q.rowStride+head*c.args.q.headStride+dim*c.args.q.dimStride ∧
    observedOffset3 v.o row head dim=row*c.args.o.rowStride+head*c.args.o.headStride+dim*c.args.o.dimStride ∧
    observedOffset4 v.k batch head row dim=batch*c.args.k.batchStride+row*c.args.k.rowStride+
      head*c.args.k.headStride+dim*c.args.k.dimStride ∧
    observedOffset4 v.v batch head row dim=batch*c.args.v.batchStride+row*c.args.v.rowStride+
      head*c.args.v.headStride+dim*c.args.v.dimStride := by
  obtain ⟨q,k,value,o⟩ := source_strides c v h
  simp only [observedOffset3,observedOffset4,q,k,value,o,List.getElem?_cons_zero,
    List.getElem?_cons_succ,Option.getD_some]
  constructor
  · trivial
  constructor
  · trivial
  constructor <;> omega

/-- Finite, executable shape/capacity/packed-interval checks. Actual source
access bounds and unique output ownership are derived by Geometry.admissible. -/
def geometryChecks (c : Host.Call) (extent : RegionBounds) : Prop :=
  0<c.args.k.heads ∧ c.args.k.heads ∣ c.args.q.heads ∧
  c.args.q.dims=c.args.k.dims ∧ c.args.k.dims=c.args.v.dims ∧
  c.args.o.rows=c.args.q.rows ∧ c.args.o.heads=c.args.q.heads ∧
  c.args.o.dims=c.args.k.dims ∧ c.args.v.heads=c.args.k.heads ∧
  c.batches=c.args.k.batches ∧ c.batches=c.args.v.batches ∧
  c.args.k.dims ∈ ({16,32,64,128,256} : Finset Nat) ∧
  (∀ b : Fin c.batches,c.length b≤c.maxInput ∧ c.start b+c.length b≤c.args.q.rows ∧
    c.prompt b+c.length b≤c.args.k.rows ∧ c.prompt b+c.length b≤c.args.v.rows) ∧
  (∀ b z : Fin c.batches,0<c.length b → 0<c.length z → b≠z →
    c.start b+c.length b≤c.start z ∨ c.start z+c.length z≤c.start b) ∧
  Geometry.Contiguous c.args.q ∧ Geometry.Contiguous c.args.o ∧
  (0<c.args.q.rows → 0<c.args.q.heads → 0<c.args.q.dims → Geometry.span3 c.args.q≤extent c.regions.query) ∧
  (0<c.args.k.batches → 0<c.args.k.rows → 0<c.args.k.heads → 0<c.args.k.dims → Geometry.span4 c.args.k≤extent c.regions.key) ∧
  (0<c.args.v.batches → 0<c.args.v.rows → 0<c.args.v.heads → 0<c.args.v.dims → Geometry.span4 c.args.v≤extent c.regions.value) ∧
  (0<c.args.o.rows → 0<c.args.o.heads → 0<c.args.o.dims → Geometry.span3 c.args.o≤extent c.regions.out) ∧
  c.batches≤extent c.regions.start ∧ c.batches≤extent c.regions.sequence ∧ c.batches≤extent c.regions.prompt

def vectorBudget (v : View) (count : Nat) : Prop :=
  v.storageOffset+(count-1)*v.strides[0]?.getD 0<2^61 ∧ v.strides[0]?.getD 0<2^61

def nativeChecks (c : Host.Call) (v : Views) : Prop :=
  c.maxInput<2^31 ∧ c.batches*c.layout.heads≤65535 ∧
  FormedBounds.Budget3 (FormedBounds.queryPadded c c.args.q) v.q.storageOffset ∧
  FormedBounds.Budget3 (FormedBounds.queryPadded c c.args.o) v.o.storageOffset ∧
  FormedBounds.Budget4 (FormedBounds.cachePadded c.args.k) v.k.storageOffset ∧
  FormedBounds.Budget4 (FormedBounds.cachePadded c.args.v) v.v.storageOffset ∧
  vectorBudget v.start c.batches ∧ vectorBudget v.sequence c.batches ∧ vectorBudget v.prompt c.batches

/-- Signed CPU metadata is compared with the same nonnegative descriptors.
Negative values cannot enter through Int.toNat totalization. -/
def metadataChecks (c : Host.Call) (start sequence prompt : Nat → Int) : Prop :=
  ∀ b : Fin c.batches,start b=(c.start b : Int) ∧ prompt b=(c.prompt b : Int) ∧
    sequence b=((c.length b+c.prompt b : Nat) : Int)

instance (a b : View) : Decidable (separate a b) := inferInstanceAs (Decidable (_ ∨ _))
instance (v : Geometry.Original.View4) : Decidable (cacheDense v) := inferInstanceAs (Decidable (_ ∨ _))
instance (v : Geometry.Original.View3) : Decidable (Geometry.Contiguous v) := inferInstanceAs (Decidable (_ ∧ _))
instance (v : Geometry.Original.View3) (off : Nat) : Decidable (FormedBounds.Budget3 v off) := inferInstanceAs (Decidable (_ ∧ _))
instance (v : Geometry.Original.View4) (off : Nat) : Decidable (FormedBounds.Budget4 v off) := inferInstanceAs (Decidable (_ ∧ _))
instance (v : View) (count : Nat) : Decidable (vectorBudget v count) := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Host.Call) (v : Views) : Decidable (tensorChecks c v) := by unfold tensorChecks all reads; infer_instance
instance (c : Host.Call) (extent : RegionBounds) : Decidable (geometryChecks c extent) := by unfold geometryChecks; infer_instance
instance (c : Host.Call) (v : Views) : Decidable (nativeChecks c v) := by unfold nativeChecks; infer_instance
instance (c : Host.Call) (start sequence prompt : Nat → Int) : Decidable (metadataChecks c start sequence prompt) := by unfold metadataChecks; infer_instance

def InputContract (c : Host.Call) (v : Views) (extent : RegionBounds)
    (start sequence prompt : Nat → Int) : Prop :=
  tensorChecks c v ∧ geometryChecks c extent ∧ nativeChecks c v ∧ metadataChecks c start sequence prompt

def checkInputs (c : Host.Call) (v : Views) (extent : RegionBounds)
    (start sequence prompt : Nat → Int) : Bool :=
  decide (tensorChecks c v) && decide (geometryChecks c extent) &&
    decide (nativeChecks c v) && decide (metadataChecks c start sequence prompt)

theorem checkInputs_iff (c : Host.Call) (v : Views) (extent : RegionBounds)
    (start sequence prompt : Nat → Int) :
    checkInputs c v extent start sequence prompt=Bool.true ↔ InputContract c v extent start sequence prompt := by
  simp only [checkInputs,InputContract,Bool.and_eq_true,decide_eq_true_eq,and_assoc]

theorem geometry (c : Host.Call) (extent : RegionBounds) (h : geometryChecks c extent) :
    Geometry.Checked c extent := by
  rcases h with ⟨kv,group,qk,kvD,outR,outH,outD,vH,kB,vB,width,metadata,packed,qContig,oContig,qSpan,kSpan,vSpan,oSpan,sSpan,lSpan,pSpan⟩
  refine ⟨kv,group,qk,kvD,outR,outH,outD,vH,le_of_eq kB,le_of_eq vB,width,?_,?_,oContig,
    qSpan,kSpan,vSpan,oSpan,sSpan,lSpan,pSpan⟩
  · intro b hb; exact metadata ⟨b,hb⟩
  · intro b z hb hz hl hk ne
    exact packed ⟨b,hb⟩ ⟨z,hz⟩ hl hk (fun eq => ne (congrArg Fin.val eq))

theorem formed (c : Host.Call) (v : Views) (h : nativeChecks c v) : FormedBounds.Checked c :=
  ⟨⟨_,h.2.2.1⟩,⟨_,h.2.2.2.1⟩,⟨_,h.2.2.2.2.1⟩,⟨_,h.2.2.2.2.2.1⟩⟩

#axiomsClean source_view_offsets
#axiomsClean checkInputs_iff
#axiomsClean geometry
#axiomsClean formed
end VeriTile.Bench.Repairs.ContextAttnFwd.Checked
