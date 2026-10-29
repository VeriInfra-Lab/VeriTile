import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.HostABI
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Contract

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
open HostABI
noncomputable section
set_option maxHeartbeats 2000000

structure Geometry where
  (B H T K V : Nat)

def tile (n : Nat) := if n≤16 then 16 else if n≤32 then 32 else 64
def shapeChecks (g : Geometry) := decide (0<g.K ∧ 0<g.V)

def checkedShape (g : Geometry) (h : shapeChecks g=Bool.true) : Shape := by
  simp only [shapeChecks,decide_eq_true_eq] at h
  refine ⟨g.B,g.H,g.T,g.K,g.V,tile g.K,tile g.V,h.1,h.2,?_,?_⟩
  all_goals unfold tile;split_ifs <;> simp

def planChecks (c : Shape) := decide (
  (∀ n∈HostABI.metadata c,n<2^31) ∧ (∀ n∈HostABI.extents c,4*n<2^63) ∧
  (0<c.heads → c.heads≤65535 ∧ c.chunks≤65535 ∧ c.valueBlocks≤65535))

theorem plan_admitted (g : Geometry) (hs : shapeChecks g=Bool.true)
    (hp : planChecks (checkedShape g hs)=Bool.true) : HostABI.Accepted (checkedShape g hs) := by
  simp only [planChecks,decide_eq_true_eq] at hp
  exact ⟨rfl,rfl,hp.1,hp.2.1,hp.2.2⟩

/-- Ordinary actual Tensor object/storage observations after the source's own
contiguous decorator. Its copies remain part of the executable host program. -/
structure Tensor where
  (object storage storageBase storageBytes storageOffset dataPointer numel : Nat)
  dimensions : List Nat
  strides : List Nat
  device : Nat
  width : Nat
  ordinary : Bool
  strided : Bool
  fp32 : Bool
  cuda : Bool
  resolved : Bool
  contiguous : Bool
  pointer : dataPointer=storageBase+width*storageOffset

def denseChecks (dims : List Nat) (device : Nat) (t : Tensor) := decide (
  t.ordinary=Bool.true ∧ t.strided=Bool.true ∧ t.fp32=Bool.true ∧ t.cuda=Bool.true ∧
  t.resolved=Bool.true ∧ t.contiguous=Bool.true ∧ t.dimensions=dims ∧ t.device=device ∧
  t.width=4 ∧ t.numel=dims.prod)

def tensorMeta (t : Tensor) : TensorMeta :=
  ⟨t.dimensions,t.strides,t.storage,t.storageOffset,t.numel,t.device,t.fp32,t.cuda,t.resolved,t.contiguous⟩

def pointerChecks (extent : Nat) (t : Tensor) := decide (
  t.width=4 ∧ 4*(t.storageOffset+t.numel)≤t.storageBytes ∧
  t.dataPointer%4=0 ∧ t.dataPointer+4*max extent t.numel<2^64)

theorem active_bytes (extent : Nat) (t : Tensor) (hp : pointerChecks extent t=Bool.true)
    (n : Nat) (hn : n<t.numel) (j : Nat) (hj : j<4) :
    t.storageBase≤t.dataPointer+4*n+j ∧ t.dataPointer+4*n+j<t.storageBase+t.storageBytes ∧
    t.dataPointer+4*n+j<2^64 := by
  simp only [pointerChecks,decide_eq_true_eq] at hp
  have he:=t.pointer
  rw [hp.1] at he
  have hm:=Nat.le_max_right extent t.numel
  omega

theorem formed_bytes (extent : Nat) (t : Tensor) (hp : pointerChecks extent t=Bool.true)
    (signed : 4*extent<2^63) (n : Nat) (hn : n<extent) (j : Nat) (hj : j<4) :
    4*n+j<2^63 ∧ t.dataPointer%4=0 ∧ t.dataPointer+4*n+j<2^64 := by
  simp only [pointerChecks,decide_eq_true_eq] at hp
  have hm:=Nat.le_max_left extent t.numel
  exact ⟨by omega,hp.2.2.1,by omega⟩

def regionShape (c : Shape) : Region→List Nat
  | .q | .k | .dq | .dk=>shapeKey c
  | .v | .incoming | .output | .dv=>shapeValue c
  | .seed | .terminal | .finalState=>shapeState c
  | .history | .dhistory=>shapeHistory c
  | .dvPart=>[c.keyBlocks,c.B,c.H,c.T,c.V]

def extent (c : Shape) : Region→Nat
  | .q | .k | .dq | .dk=>keyExtent c
  | .v | .incoming | .output | .dv=>valueExtent c
  | .seed | .terminal | .finalState=>stateExtent c
  | .history | .dhistory=>historyExtent c
  | .dvPart=>partialExtent c

theorem region_count (c : Shape) (r : Region) : (regionShape c r).prod=r.capacity c := by
  cases r <;> simp [regionShape,Region.capacity,shapeKey,shapeValue,shapeState,shapeHistory,
    Shape.keySize,Shape.valueSize,Shape.stateSize,Shape.historySize,Shape.partialSize,
    Shape.keyHead,Shape.valueHead,Shape.stateHead,Shape.historyHead,Shape.heads,Nat.mul_assoc]

def inputChecks (c : Shape) (q k v : Tensor) (seed : Option Tensor) :=
  denseChecks (shapeKey c) q.device q && denseChecks (shapeKey c) q.device k &&
  denseChecks (shapeValue c) q.device v &&
    (match seed with | none=>Bool.true | some t=>denseChecks (shapeState c) q.device t)

theorem dense_admitted (dims : List Nat) (device : Nat) (t : Tensor)
    (h : denseChecks dims device t=Bool.true) :
    (tensorMeta t).dimensions=dims ∧ (tensorMeta t).fp32=Bool.true ∧ (tensorMeta t).cuda=Bool.true ∧
    (tensorMeta t).resolved=Bool.true ∧ (tensorMeta t).contiguous=Bool.true ∧ (tensorMeta t).device=device := by
  simp only [denseChecks,decide_eq_true_eq] at h
  exact ⟨h.2.2.2.2.2.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2.2.1⟩

#axiomsClean plan_admitted
#axiomsClean active_bytes
#axiomsClean formed_bytes
#axiomsClean region_count
#axiomsClean dense_admitted
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
