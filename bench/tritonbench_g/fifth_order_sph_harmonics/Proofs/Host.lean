import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Physical
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Access

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Host
open VeriTile.Triton FP.Structural Packing Storage
noncomputable section
set_option maxHeartbeats 3000000

/-- Object identity and metadata observed at entry. These fields are query
observations used by the guard, not claims that autograd leaves flags/version unchanged. -/
structure TensorObject where
  identity : Nat
  metadata : TensorView
  version : Nat

/-- Ordinary successful torch.empty metadata/freshness. No numerical output
or kernel behavior is a field of this allocator interface. -/
structure FreshOutput (x : TensorView) (y : TensorObject) : Prop where
  kind : TensorKind y.metadata
  rows : y.metadata.rows=x.rows
  columns : y.metadata.columns=11
  strides : y.metadata.strides=[11,1]
  device : y.metadata.tensor.device=x.tensor.device
  grad : y.metadata.requiresGrad=Bool.false
  bounds : TensorBounds y.metadata
  apart : 0<x.rows → x.pointer+x.rows*12≤y.metadata.pointer ∨
    y.metadata.pointer+x.rows*44≤x.pointer

structure Request where
  coordinates : TensorObject
  supplied : Option TensorObject
  block : Int
  column : Int
  argumentCount : Nat
  arity : 1≤argumentCount ∧ argumentCount≤5

def Request.output (q : Request) (allocated : TensorObject) := q.supplied.getD allocated

def Admitted (q : Request) : Prop :=
  ForwardContract q.coordinates.metadata (q.supplied.map TensorObject.metadata) q.block q.column

/-- An allocator result is used only when the caller omitted the destination. -/
def Allocation (q : Request) (allocated : TensorObject) : Prop :=
  q.supplied=none → FreshOutput q.coordinates.metadata allocated

theorem destination (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) :
    Destination q.coordinates.metadata (q.output allocated).metadata q.column := by
  unfold Admitted ForwardContract at admit
  cases h:q.supplied with
  | some y=>
    simp only [h,Option.map_some] at admit
    simpa [Request.output,h] using admit.2.2.2.1
  | none=>
    simp only [h,Option.map_none] at admit
    have hc:q.column=0 := admit.2.2.2.1
    have fresh:=allocation h
    have width:=fresh.columns
    simp only [Request.output,h,Option.getD_none,Destination,hc,Int.toNat_zero,Nat.zero_add]
    exact ⟨fresh.kind,fresh.rows,by omega,by omega,
      by simpa [fresh.columns] using fresh.strides,fresh.device,fresh.grad,fresh.bounds,
      by simpa [fresh.columns,Nat.mul_assoc] using fresh.apart⟩

def parameters (q : Request) (admit : Admitted q) (allocated : TensorObject) : Parameters where
  rows:=q.coordinates.metadata.rows
  blockSize:=q.block.toNat
  blockPositive:=by have hb:=admit.2.1.1;omega
  outputWidth:=(q.output allocated).metadata.columns
  columnOffset:=q.column.toNat
  upstreamRowStride:=0
  upstreamColStride:=0

theorem parameters_fit (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) :
    (parameters q admit allocated).columnOffset+11≤(parameters q admit allocated).outputWidth :=
  (destination q admit allocated allocation).2.2.1

theorem parameters_launch (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) :
    LaunchContract (parameters q admit allocated).rows (parameters q admit allocated).outputWidth
      (parameters q admit allocated).blockSize (parameters q admit allocated).columnOffset := by
  have detail:=admit.2.2.2
  cases h:q.supplied with
  | some y=>
    simp only [h,Option.map_some] at detail
    simpa [parameters,Request.output,h] using detail.2
  | none=>
    simp only [h,Option.map_none] at detail
    have width:allocated.metadata.columns=11 := (allocation h).columns
    simpa [parameters,Request.output,h,width] using detail.2

def forwardBuffers (q : Request) (allocated : TensorObject) : Physical.Buffers :=
  ⟨q.coordinates.metadata.pointer,0,(q.output allocated).metadata.pointer⟩

structure Context where
  savedCoordinates : TensorObject
  outputWidth : Nat
  blockSize : Nat
  columnOffset : Nat
  argumentCount : Nat

def save (q : Request) (allocated : TensorObject) : Context :=
  ⟨q.coordinates,(q.output allocated).metadata.columns,q.block.toNat,q.column.toNat,q.argumentCount⟩

def loadSaved (ctx : Context) (currentVersion : Nat) : Except String TensorObject :=
  if currentVersion=ctx.savedCoordinates.version then .ok ctx.savedCoordinates
  else .error "saved coordinate tensor was modified in place"

def backwardResults (ctx : Context) (gradient : TensorObject) : List (Option TensorObject) :=
  ([some gradient,none,none,none,none] : List (Option TensorObject)).take ctx.argumentCount

theorem saved_identity (q : Request) (allocated : TensorObject) :
    (save q allocated).savedCoordinates=q.coordinates := rfl

theorem saved_version (ctx : Context) (currentVersion : Nat) :
    loadSaved ctx currentVersion=.ok ctx.savedCoordinates ↔ currentVersion=ctx.savedCoordinates.version := by
  simp [loadSaved]

theorem backward_arity (q : Request) (allocated gradient : TensorObject) :
    (backwardResults (save q allocated) gradient).length=q.argumentCount := by
  simp only [backwardResults,save,List.length_take]
  simp
  exact q.arity.2

theorem backward_identity (q : Request) (allocated gradient : TensorObject) :
    (backwardResults (save q allocated) gradient).head?=some (some gradient) := by
  cases h:q.argumentCount with
  | zero=>have arity:=q.arity.1;omega
  | succ n=>simp [backwardResults,save,h]

/-- Ordinary version-indexed Tensor contents. A successful saved-tensor load
selects the exact identity/version saved by forward, without a coordinate copy. -/
theorem saved_contents {A : Type} (history : Nat → Nat → Nat → A) (q : Request)
    (allocated : TensorObject) (currentVersion : Nat)
    (loaded : loadSaved (save q allocated) currentVersion=.ok q.coordinates) :
    history (save q allocated).savedCoordinates.identity currentVersion=
      history q.coordinates.identity q.coordinates.version := by
  have same : currentVersion=q.coordinates.version := by
    exact (saved_version (save q allocated) currentVersion).mp loaded
  simp only [save,same]

theorem returned_metadata (q : Request) (admit : Admitted q)
    (allocated : TensorObject) (allocation : Allocation q allocated) :
    (q.output allocated).metadata.tensor.dtype=.floating .fp32 ∧
    (q.output allocated).metadata.tensor.shape=[q.coordinates.metadata.rows,(q.output allocated).metadata.columns] ∧
    (q.output allocated).metadata.tensor.device=q.coordinates.metadata.tensor.device := by
  have d:=destination q admit allocated allocation
  have kind:=d.1
  unfold TensorKind at kind
  refine ⟨kind.2.1,?_,d.2.2.2.2.2.1⟩
  simpa [d.2.1] using kind.2.2.2.2.2

/-- The device guard saves and restores the prior device around the same
coordinate-device launch; it does not narrow admission to the current GPU. -/
def deviceGuard (current requested : Nat) (action : Nat → α) : α × Nat := (action requested,current)

def forward (M : Algebra Word) (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (heap : Heap) : TensorObject × Context × Heap :=
  (q.output allocated,save q allocated,
    (Physical.launch M .forward (parameters q admit allocated) (forwardBuffers q allocated) heap).bytes)

theorem supplied_identity (M : Algebra Word) (q : Request) (admit : Admitted q)
    (allocated y : TensorObject) (supplied : q.supplied=some y) (heap : Heap) :
    (forward M q admit allocated heap).1.identity=y.identity := by simp [forward,Request.output,supplied]

def backwardParameters (p : Parameters) (upstream : TensorView) : Parameters :=
  {p with upstreamRowStride:=upstream.rowStride.toNat,upstreamColStride:=upstream.colStride.toNat}

def backwardBuffers (ctx : Context) (upstream gradient : TensorObject) : Physical.Buffers :=
  ⟨ctx.savedCoordinates.metadata.pointer,upstream.metadata.pointer,gradient.metadata.pointer⟩

/-- Fresh zeros_like: every byte of the dense [rows,3] gradient is initialized;
bytes outside that successful fresh allocation are untouched. -/
def zeroGradient (pointer rows : Nat) (heap : Heap) : Heap := fun q=>
  if pointer≤q ∧ q<pointer+12*rows then 0 else heap q

theorem zeroGradient_read (pointer rows : Nat) (heap : Heap) (r c : Nat)
    (hr : r<rows) (hc : c<3) : read (zeroGradient pointer rows heap) (pointer+4*(r*3+c))=fun _=>0 := by
  funext k
  have hk:=k.isLt
  simp [Storage.read,zeroGradient,show pointer≤pointer+4*(r*3+c)+k.val ∧
    pointer+4*(r*3+c)+k.val<pointer+12*rows by omega]

/-- Successful zeros_like allocation observations; allocation disjointness is
about actual storage intervals, not Tensor identity tags. -/
structure FreshGradient (x upstream : TensorView) (gradient : TensorObject) : Prop where
  kind : TensorKind gradient.metadata
  rows : gradient.metadata.rows=x.rows
  columns : gradient.metadata.columns=3
  strides : gradient.metadata.strides=[3,1]
  device : gradient.metadata.tensor.device=x.tensor.device
  bounds : TensorBounds gradient.metadata
  coordinatesApart : gradient.metadata.storageBase+gradient.metadata.capacity≤x.storageBase ∨
    x.storageBase+x.capacity≤gradient.metadata.storageBase
  upstreamApart : gradient.metadata.storageBase+gradient.metadata.capacity≤upstream.storageBase ∨
    upstream.storageBase+upstream.capacity≤gradient.metadata.storageBase

def backward (M : Algebra Word) (p : Parameters) (ctx : Context) (upstream gradient : TensorObject)
    (heap : Heap) : List (Option TensorObject) × Heap :=
  (backwardResults ctx gradient,
    (Physical.launch M .backward (backwardParameters p upstream.metadata)
      (backwardBuffers ctx upstream gradient) (zeroGradient gradient.metadata.pointer p.rows heap)).bytes)

#axiomsClean destination
#axiomsClean parameters_fit
#axiomsClean parameters_launch
#axiomsClean saved_identity
#axiomsClean saved_version
#axiomsClean saved_contents
#axiomsClean returned_metadata
#axiomsClean backward_arity
#axiomsClean backward_identity
#axiomsClean supplied_identity
#axiomsClean zeroGradient_read
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Host
