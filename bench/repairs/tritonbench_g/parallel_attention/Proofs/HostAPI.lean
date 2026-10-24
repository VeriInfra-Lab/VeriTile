import bench.repairs.tritonbench_g.parallel_attention.Proofs.Contract
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Types

namespace VeriTile.Bench.Repairs.ParallelAttention.HostAPI
noncomputable section
open scoped BigOperators

/-- A returned Tensor handle includes its actual storage pointer. A storage
offset is included in `base`; allocation/metadata are ordinary Tensor contracts. -/
structure Tensor where
  base : Nat
  metadata : TensorMetadata

def tensor (s : Shape) (kind : OutputKind) (base : Nat) : Tensor :=
  ⟨base,outputMetadata s kind⟩

def Tensor.read (h : Tensor) (m : Memory α) (bh t e : Nat) : α :=
  m (h.base+dense h.metadata.time h.metadata.features bh t e)

structure Inputs where
  q : Tensor
  k : Tensor
  v : Tensor

def Inputs.Valid (x : Inputs) (s : Shape) : Prop :=
  x.q.metadata=outputMetadata s .query ∧
  x.k.metadata=outputMetadata s .key ∧ x.v.metadata=outputMetadata s .value

/-- The unused forward do/dz slots alias the existing V allocation. They
neither name future allocations nor constrain future cotangent values. -/
def forwardBases (x : Inputs) : Bases :=
  ⟨x.q.base,x.k.base,x.v.base,x.v.base,x.v.base⟩

def inputValues (a : Ops α) (s : Shape) (x : Inputs) (m : Memory α) (bh : Nat) :=
  headInputs (currentInputs a s (forwardBases x) m) bh

/-- These are exactly the three saved tensors and the scale selected at this
forward call. No backward argument is a field or a forward-call premise. -/
structure Saved (α : Type) where
  tensors : Inputs
  shape : Shape
  scale : α

def save (a : Ops α) (s : Shape) (x : Inputs) : Saved α :=
  ⟨x,s,a.scale s.K s.useScale⟩

def Saved.kernelContext (ctx : Saved α) : SavedContext :=
  ⟨forwardBases ctx.tensors,ctx.shape⟩

abbrev Empty (s : Shape) : Prop := s.B*s.H*s.T=0

/-- Empty dispatch returns empty allocated handles but launches no kernels. -/
def dispatch (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) : Memory α :=
  if Empty s then m else combinedRun a p s b trees scratch output ks hs m

theorem events_empty (s : Shape) (empty : Empty s) (scratch : Bool)
    (kinds : List OutputKind) : events s scratch kinds=[] := by
  have hc (kind : OutputKind) : kind.cells s=0 := by
    simp [OutputKind.cells,Empty] at *
    simp [empty]
  have hn (kind : OutputKind) : kind.count s=0 := by
    change kind.shards s*(s.B*s.H*s.T*kind.width s)=0
    simp [show s.B*s.H*s.T=0 from empty]
  cases scratch <;> simp [events,hc,hn]

theorem dispatch_eq (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (kinds : List OutputKind) (ks hs : List Event)
    (layout : RunLayout s b trees scratch output kinds ks hs) (m : Memory α) :
    dispatch a p s b trees scratch output ks hs m=
      combinedRun a p s b trees scratch output ks hs m := by
  unfold dispatch
  split_ifs with h
  · have hk : ks=[] := List.perm_nil.mp (layout.kernelCoverage.trans
      (List.Perm.of_eq (events_empty s h true kinds)))
    have hh : hs=[] := List.perm_nil.mp (layout.hostCoverage.trans
      (List.Perm.of_eq (events_empty s h false kinds)))
    simp [hk,hh,combinedRun,combinedHost,combinedScratch,runTagged]
  · rfl

inductive Returned where
  | both (output normalizer : Tensor)
  | single (output : Tensor)

def Returned.read (r : Returned) (m : Memory α) (bh : Nat) : PublicResult α :=
  match r with
  | .both o z => .both (o.read m bh) (fun t => z.read m bh t 0)
  | .single o => .single (o.read m bh)

def returned (s : Shape) (output : OutputKind → Nat) (normalized : Nat)
    (returnBoth normalize : Bool) : Returned :=
  if returnBoth then .both (tensor s .forward (output .forward)) (tensor s .normalizer (output .normalizer))
  else .single (tensor s .forward (if normalize then normalized else output .forward))

def forwardMemory (a : Ops α) (p : Plans) (s : Shape) (x : Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (eps : α) (normalized : Nat)
    (returnBoth normalize : Bool) : Memory α :=
  let raw := dispatch a p s (forwardBases x) trees scratch output ks hs m
  if returnBoth then raw else if normalize then
    normalizeRun a eps (s.B*s.H) s.T s.V (output .forward) (output .normalizer) normalized raw
  else raw

structure ForwardResult (α : Type) where
  memory : Memory α
  saved : Saved α
  returned : Returned
  rawOutput : Tensor
  rawNormalizer : Tensor

def forward (a : Ops α) (p : Plans) (s : Shape) (x : Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (eps : α) (normalized : Nat)
    (returnBoth normalize : Bool) : ForwardResult α :=
  ⟨forwardMemory a p s x trees scratch output ks hs m eps normalized returnBoth normalize,
    save a s x,returned s output normalized returnBoth normalize,
    tensor s .forward (output .forward),tensor s .normalizer (output .normalizer)⟩

/-- Only normalized-output allocation separation; no numerical premise. -/
def NormalizationLayout (s : Shape) (output : OutputKind → Nat) (normalized : Nat) : Prop :=
  ∀ i,i<s.B*s.H*s.T*s.V → normalized+i∉
    normalizeReads (s.B*s.H) s.T s.V (output .forward) (output .normalizer)

/-- The public reader follows the actual returned handle, including the new
normalization allocation. Return-both takes priority over normalization. -/
theorem forward_saved (a : Ops α) (p : Plans) (s : Shape) (x : Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (eps : α) (normalized : Nat) (returnBoth normalize : Bool) :
    (forward a p s x trees scratch output ks hs m eps normalized returnBoth normalize).saved=save a s x := rfl

theorem forward_frame (a : Ops α) (p : Plans) (s : Shape) (x : Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (eps : α) (normalized : Nat) (returnBoth normalize : Bool)
    (address : Nat) (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address)
    (outsideHost : ∀ ev∈hs,tensorAddress output ev≠address)
    (outsideNormalized : normalize=true → returnBoth=false →
      address<normalized ∨ normalized+s.B*s.H*s.T*s.V≤address) :
    (forward a p s x trees scratch output ks hs m eps normalized returnBoth normalize).memory address=m address := by
  have raw : dispatch a p s (forwardBases x) trees scratch output ks hs m address=m address := by
    unfold dispatch
    split_ifs
    · rfl
    · exact combined_frame a p s (forwardBases x) trees scratch output ks hs m address outsideKernel outsideHost
  cases returnBoth <;> cases normalize <;>
    simp only [forward,forwardMemory,Bool.false_eq_true,if_false,if_true]
  · exact raw
  · rw [normalize_frame a eps (s.B*s.H) s.T s.V (output .forward) (output .normalizer)
      normalized _ address (by simpa [Nat.mul_assoc] using outsideNormalized rfl rfl)]
    exact raw
  · exact raw
  · exact raw

/-- Incoming cotangents may be expanded or strided tensors. `base` includes
storage offset; zero strides cover the ordinary expanded-view case. -/
structure View where
  base : Nat
  batchStride : Nat
  headStride : Nat
  timeStride : Nat
  featureStride : Nat

def View.address (v : View) (s : Shape) (bh t e : Nat) : Nat :=
  v.base+(bh/s.H)*v.batchStride+(bh%s.H)*v.headStride+t*v.timeStride+e*v.featureStride

def viewRead (a : Ops α) (s : Shape) (D : Nat) (v : View) (m : Memory α) (bh t e : Nat) : α :=
  if bh<s.B*s.H ∧ t<s.T ∧ e<D then m (v.address s bh t e) else a.zero

/-- The ordinary `.contiguous()` contract copies active logical elements.
It mentions only input values and storage metadata, never any gradient result. -/
structure Materialized (s : Shape) (kind : OutputKind) (incoming ready : Memory α)
    (v : View) (h : Tensor) : Prop where
  metadata : h.metadata=outputMetadata s kind
  values : ∀ bh t e,bh<s.B*s.H → t<s.T → e<kind.width s →
    ready (h.base+dense s.T (kind.width s) bh t e)=incoming (v.address s bh t e)

def inputCellsOnly (s : Shape) (x : Inputs) : Set Nat :=
  tensorCells x.q.base (s.B*s.H) s.T s.K ∪ tensorCells x.k.base (s.B*s.H) s.T s.K ∪
  tensorCells x.v.base (s.B*s.H) s.T s.V

/-- Saved Tensor/version preservation is imposed only when backward is called.
Forward has no obligation to predict the later cotangent or allocator state. -/
def SavedValid (ctx : Saved α) (forwardEntry ready : Memory α) : Prop :=
  Set.EqOn ready forwardEntry (inputCellsOnly ctx.shape ctx.tensors)

def backwardBases (ctx : Saved α) (do_ dz : Tensor) : Bases :=
  let b := ctx.kernelContext.inputs
  ⟨b.q,b.k,b.v,do_.base,dz.base⟩

def incomingCotangent (a : Ops α) (s : Shape) (incoming : Memory α)
    (do_ dz : View) (bh : Nat) : Cotangent α :=
  ⟨viewRead a s s.V do_ incoming bh,fun t => viewRead a s 1 dz incoming bh t 0⟩

theorem saved_inputs (a : Ops α) (ctx : Saved α) (forwardEntry ready : Memory α)
    (valid : SavedValid ctx forwardEntry ready) (do_ dz : Tensor) (bh : Nat) :
    headInputs (currentInputs a ctx.shape (backwardBases ctx do_ dz) ready) bh=
      inputValues a ctx.shape ctx.tensors forwardEntry bh := by
  have hq := tensorRead_congr a ready forwardEntry ctx.tensors.q.base
    (ctx.shape.B*ctx.shape.H) ctx.shape.T ctx.shape.K (by intro i hi;exact valid (Or.inl (Or.inl hi)))
  have hk := tensorRead_congr a ready forwardEntry ctx.tensors.k.base
    (ctx.shape.B*ctx.shape.H) ctx.shape.T ctx.shape.K (by intro i hi;exact valid (Or.inl (Or.inr hi)))
  have hv := tensorRead_congr a ready forwardEntry ctx.tensors.v.base
    (ctx.shape.B*ctx.shape.H) ctx.shape.T ctx.shape.V (by intro i hi;exact valid (Or.inr hi))
  simp only [headInputs,currentInputs,backwardBases,Saved.kernelContext,forwardBases,inputValues]
  rw [hq,hk,hv]

theorem materialized_cotangent (a : Ops α) (ctx : Saved α) (incoming ready : Memory α)
    (doView dzView : View) (do_ dz : Tensor)
    (ho : Materialized ctx.shape .forward incoming ready doView do_)
    (hz : Materialized ctx.shape .normalizer incoming ready dzView dz) (bh : Nat) :
    headCotangent (currentInputs a ctx.shape (backwardBases ctx do_ dz) ready) bh=
      incomingCotangent a ctx.shape incoming doView dzView bh := by
  unfold headCotangent incomingCotangent
  congr 1
  · funext t e
    change tensorRead a ready do_.base (ctx.shape.B*ctx.shape.H) ctx.shape.T ctx.shape.V bh t e=_
    unfold tensorRead viewRead
    split_ifs with h
    · exact ho.values bh t e h.1 h.2.1 h.2.2
    · rfl
  · funext t
    change tensorRead a ready dz.base (ctx.shape.B*ctx.shape.H) ctx.shape.T 1 bh t 0=_
    unfold tensorRead viewRead
    split_ifs with h
    · exact hz.values bh t 0 h.1 h.2.1 h.2.2
    · rfl

structure Gradients where
  q : Tensor
  k : Tensor
  v : Tensor
  scale : Option Tensor

def gradients (s : Shape) (output : OutputKind → Nat) : Gradients :=
  ⟨tensor s .query (output .query),tensor s .key (output .key),tensor s .value (output .value),none⟩

structure BackwardResult (α : Type) where
  memory : Memory α
  returned : Gradients

/-- Backward consumes the scale saved by the actual forward call. Other
scalar ports remain unchanged; only the selected shape/flag call is rebound. -/
def savedOps (a : Ops α) (ctx : Saved α) : Ops α :=
  { a with scale := fun n flag => if n=ctx.shape.K ∧ flag=ctx.shape.useScale then ctx.scale else a.scale n flag }

theorem saved_ops_eq (a : Ops α) (s : Shape) (x : Inputs) : savedOps a (save a s x)=a := by
  unfold savedOps save
  congr 1
  funext n flag
  split_ifs with h
  · rw [h.1,h.2]
  · rfl

def backward (a : Ops α) (p : Plans) (ctx : Saved α) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (ready : Memory α) : BackwardResult α :=
  ⟨dispatch (savedOps a ctx) p ctx.shape (backwardBases ctx do_ dz) trees scratch output ks hs ready,
    gradients ctx.shape output⟩

theorem backward_frame (a : Ops α) (p : Plans) (ctx : Saved α) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (ready : Memory α) (address : Nat)
    (outsideKernel : ∀ ev∈ks,tensorAddress scratch ev≠address)
    (outsideHost : ∀ ev∈hs,tensorAddress output ev≠address) :
    (backward a p ctx do_ dz trees scratch output ks hs ready).memory address=ready address := by
  dsimp only [backward]
  unfold dispatch
  split_ifs
  · rfl
  · exact combined_frame (savedOps a ctx) p ctx.shape (backwardBases ctx do_ dz)
      trees scratch output ks hs ready address outsideKernel outsideHost

theorem empty_backward (a : Ops α) (p : Plans) (ctx : Saved α) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (ready : Memory α) (empty : Empty ctx.shape) :
    (backward a p ctx do_ dz trees scratch output ks hs ready).memory=ready ∧
    (backward a p ctx do_ dz trees scratch output ks hs ready).returned.scale=none := by
  constructor
  · exact if_pos empty
  · rfl

/-- Ordinary fresh allocation conditions preserve the actual saved Q/K/V
storage at forward return. This theorem still has no backward argument. -/
theorem forward_preserves_saved (a : Ops α) (p : Plans) (s : Shape) (x : Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (entry : Memory α) (eps : α) (normalized : Nat)
    (returnBoth normalize : Bool)
    (kernelFresh : ∀ ev∈ks,tensorAddress scratch ev∉inputCellsOnly s x)
    (hostFresh : ∀ ev∈hs,tensorAddress output ev∉inputCellsOnly s x)
    (normalizeFresh : normalize=true → returnBoth=false →
      ∀ i,i<s.B*s.H*s.T*s.V → normalized+i∉inputCellsOnly s x) :
    SavedValid (save a s x) entry
      (forward a p s x trees scratch output ks hs entry eps normalized returnBoth normalize).memory := by
  intro address ha
  apply forward_frame
  · intro ev he eq
    exact kernelFresh ev he (eq ▸ ha)
  · intro ev he eq
    exact hostFresh ev he (eq ▸ ha)
  · intro hn hb
    by_contra outside
    have lower : normalized≤address := by omega
    have upper : address<normalized+s.B*s.H*s.T*s.V := by omega
    have hi : address-normalized<s.B*s.H*s.T*s.V := by omega
    exact normalizeFresh hn hb (address-normalized) hi (by
      have eq : normalized+(address-normalized)=address := by omega
      simpa only [eq] using ha)

/-- The core autograd backward reads the context actually returned by a
previous forward call. The later cotangents appear only in this theorem. -/
theorem empty_forward (a : Ops α) (p : Plans) (s : Shape) (x : Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (entry : Memory α) (eps : α) (normalized : Nat)
    (returnBoth normalize : Bool) (empty : Empty s) :
    (forward a p s x trees scratch output ks hs entry eps normalized returnBoth normalize).memory=entry := by
  have noDispatch : dispatch a p s (forwardBases x) trees scratch output ks hs entry=entry := if_pos empty
  have size : s.B*s.H*s.T*s.V=0 := by rw [empty,zero_mul]
  cases returnBoth <;> cases normalize <;>
    simp only [forward,forwardMemory,Bool.false_eq_true,if_false,if_true,noDispatch,normalizeRun,size,List.range_zero,runStores]


end
end VeriTile.Bench.Repairs.ParallelAttention.HostAPI
