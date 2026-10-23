import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.HostConfig
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Allocation

/-! Shared initial caller objects, request-derived configuration and ordinary
coordinate copy contracts. No Real or FP execution is imported. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostRequest
open VeriTile.Triton HostFP
noncomputable section

structure Tensor (α : Type) where
  objectId : Nat
  allocationId : Nat
  shape : List Nat
  dtype : NativeSource.Storage
  device : Nat
  address : Nat → Nat
  action : α → α
  involution : Function.Involutive action

def Tensor.logical {α : Type} (t : Tensor α) (memory : Nat → α) (i : Nat) :=
  t.action (memory (t.address i))

structure Inner (α : Type) where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  gateK : Bool
  q : Tensor α
  key : Tensor α
  value : Tensor α
  gate : Tensor α
  seed : Option (Tensor α)
  final : Option (Tensor α)
  before : Nat → α

structure Regions where
  key : RegionName
  value : RegionName
  gate : RegionName
  history : RegionName
  seed : RegionName
  final : RegionName
  distinct : [key,value,gate,history,seed,final].Nodup

def optionalDtype {α : Type} (t : Option (Tensor α)) : FloatDType :=
  (t.map (fun x => x.dtype.dtype)).getD .fp32

def Inner.config {α : Type} (x : Inner α) (r : Regions) : HostFP.Inner where
  dot := x.key.dtype
  gate := x.gate.dtype
  history := x.q.dtype
  seed := optionalDtype x.seed
  final := optionalDtype x.final
  k := r.key
  v := r.value
  g := r.gate
  h := r.history
  h0 := r.seed
  ht := r.final
  heads := x.B*x.H
  T := x.T
  K := x.K
  V := x.V
  BT := x.BT
  BK := x.BK
  BV := x.BV
  gateK := x.gateK
  hasSeed := x.seed.isSome
  storeFinal := x.final.isSome

/-- Exactly the accepted shape/device/dtype role conditions. Power-of-two
blocks and lower bound reproduce validation without restricting logical tails. -/
structure Inner.Valid {α : Type} (x : Inner α) : Prop where
  keyShape : x.key.shape=[x.B,x.H,x.T,x.K]
  valueShape : x.value.shape=[x.B,x.H,x.T,x.V]
  gateShape : x.gate.shape=[x.B,x.H,x.T,if x.gateK then x.K else x.V]
  seedShape : ∀ t, x.seed=some t → t.shape=[x.B,x.H,x.K,x.V]
  finalShape : ∀ t, x.final=some t → t.shape=[x.B,x.H,x.K,x.V]
  dotDtype : x.key.dtype=x.value.dtype
  devices : x.key.device=x.q.device ∧ x.value.device=x.q.device ∧ x.gate.device=x.q.device
  optionalDevices : ∀ t, x.seed=some t ∨ x.final=some t → t.device=x.q.device
  bt : 16≤x.BT
  bk : 16≤x.BK
  bv : 16≤x.BV
  blocks : (∃ n, x.BT=2^n) ∧ (∃ n, x.BK=2^n) ∧ (∃ n, x.BV=2^n)
  outputInjective : ∀ t, x.final=some t →
    ∀ i j, i<x.B*x.H*x.K*x.V → j<x.B*x.H*x.K*x.V → t.address i=t.address j → i=j


inductive PreparationStage where
  | key | value | gate | repeatGate | seed | allocateFinal | allocateHistory
  deriving DecidableEq

def preparationOrder (hasSeed hasFinal : Bool) : List PreparationStage :=
  [.key,.value,.gate,.repeatGate] ++ (if hasSeed then [.seed] else []) ++
    (if hasFinal then [.allocateFinal] else []) ++ [.allocateHistory]


structure Returned (α : Type) where
  objectId : Nat
  allocationId : Nat
  shape : List Nat
  dtype : NativeSource.Storage
  values : Nat → α
  caller : Nat → α

/-- Returned contiguous tensor placement is the ordinary successful final
slice/reshape/contiguous result. The caller output remains its original object. -/
structure ReturnPlacement {α : Type} (x : Inner α) where
  objectId : Nat
  allocationId : Nat
  freshCaller : ∀ t∈[some x.q,some x.key,some x.value,some x.gate,x.seed,x.final],
    ∀ tensor, t=some tensor → allocationId≠tensor.allocationId


/-- Tensor copy_ stores through the original view's involution. Since the
kernel final buffer already has ht's dtype, this same-dtype copy adds no cast. -/
def copyFinal {α : Type} (x : Inner α) (values : Nat → α) : Nat → α := fun addr =>
  match x.final with
  | none => x.before addr
  | some t => if h : ∃ i, i<x.B*x.H*x.K*x.V ∧ t.address i=addr then
      t.action (values (Classical.choose h)) else x.before addr


theorem copy_frame {α : Type} (x : Inner α) (values : Nat → α) (addr : Nat)
    (h : ∀ t, x.final=some t → ¬∃ i, i<x.B*x.H*x.K*x.V ∧ t.address i=addr) :
    copyFinal x values addr=x.before addr := by
  cases ht : x.final with
  | none => simp [copyFinal,ht]
  | some t => simp [copyFinal,ht,h t ht]

theorem copy_read {α : Type} (x : Inner α) (hx : x.Valid) (values : Nat → α)
    (t : Tensor α) (ht : x.final=some t) (i : Nat) (hi : i<x.B*x.H*x.K*x.V) :
    t.logical (copyFinal x values) i=values i := by
  have he : ∃ j, j<x.B*x.H*x.K*x.V ∧ t.address j=t.address i := ⟨i,hi,rfl⟩
  have hj := Classical.choose_spec he
  have eq : Classical.choose he=i := hx.outputInjective t ht _ i hj.1 hi hj.2
  simp only [Tensor.logical,copyFinal,ht,dif_pos he,eq]
  exact t.involution (values i)


/-- Successful ordinary allocations, independent of any numerical interpretation. -/
structure Allocated {α : Type} (x : Inner α) (r : Regions) where
  buffers : RegionName → Allocation.Buffer
  keyExtent : (x.config r).heads*(x.config r).PT*(x.config r).PK≤(buffers r.key).extent
  valueExtent : (x.config r).heads*(x.config r).PT*(x.config r).PV≤(buffers r.value).extent
  gateExtent : (x.config r).heads*(x.config r).PT*(if x.gateK then (x.config r).PK else (x.config r).PV)≤(buffers r.gate).extent
  historyExtent : (x.config r).heads*(x.config r).NT*(x.config r).PK*(x.config r).PV≤(buffers r.history).extent
  seedExtent : x.seed.isSome=Bool.true → (x.config r).heads*(x.config r).PK*(x.config r).PV≤(buffers r.seed).extent
  finalExtent : x.final.isSome=Bool.true → (x.config r).heads*(x.config r).PK*(x.config r).PV≤(buffers r.final).extent
  keyDtype : (buffers r.key).dtype=x.key.dtype
  valueDtype : (buffers r.value).dtype=x.value.dtype
  gateDtype : (buffers r.gate).dtype=x.gate.dtype
  historyDtype : (buffers r.history).dtype=x.q.dtype
  seedDtype : ∀ t, x.seed=some t → (buffers r.seed).dtype=t.dtype
  finalDtype : ∀ t, x.final=some t → (buffers r.final).dtype=t.dtype
  privateBuffers : ∀ rg∈[r.key,r.value,r.gate,r.history,r.seed,r.final],
    ∀ t∈[some x.q,some x.key,some x.value,some x.gate,x.seed,x.final],
      ∀ tensor, t=some tensor → (buffers rg).allocationId≠tensor.allocationId
  disjoint : ∀ a∈[r.key,r.value,r.gate,r.history,r.seed,r.final],
    ∀ b∈[r.key,r.value,r.gate,r.history,r.seed,r.final], a≠b → (buffers a).allocationId≠(buffers b).allocationId

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostRequest
