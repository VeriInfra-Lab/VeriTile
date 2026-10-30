import bench.tritonbench_g.attn_fwd_triton.Proofs.ByteLayout
import bench.tritonbench_g.attn_fwd_triton.Proofs.HostBlock

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton
open Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd.Address (signed64 wrap64 signed64_exact wrap64_exact)
set_option maxHeartbeats 1000000

structure Strides where
  (batch head row dim : Nat)
  deriving DecidableEq
structure Geometry where
  (batches heads context headDim : Nat)
  stride : Role→Strides

def Strides.at (s : Strides) (z h n d : Nat) := z*s.batch+h*s.head+n*s.row+d*s.dim
def Strides.raw (s : Strides) : List Int := [s.batch,s.head,s.row,s.dim]
def countQ (g : Geometry) := (g.context+127)/128
def countK (g : Geometry) := (g.context+63)/64

def span (g : Geometry) (r : Role) :=
  match r with
  | .queryScale=>g.batches*g.heads*countQ g
  | .keyScale=>g.batches*g.heads*countK g
  | _=>if g.batches=0 ∨ g.heads=0 ∨ g.context=0 ∨ g.headDim=0 then 0 else
    (g.batches-1)*(g.stride r).batch+(g.heads-1)*(g.stride r).head+
    (g.context-1)*(g.stride r).row+(g.headDim-1)*(g.stride r).dim+1

def extent (g : Geometry) (r : Role) :=
  g.batches*(g.stride r).batch+g.heads*(g.stride r).head+
    (g.context+256)*(g.stride r).row+HostShape.blockD g.headDim*(g.stride r).dim+1

def envelope (g : Geometry) :=
  g.batches+g.heads+g.context+g.headDim+HostShape.blockD g.headDim+512+g.batches*g.heads+
  extent g .query+extent g .key+extent g .value+extent g .output+
  g.batches*g.heads*countQ g+countQ g+2+g.batches*g.heads*countK g+countK g+2

def roles : List Role := [.query,.key,.value,.output,.queryScale,.keyScale]

/-- These are ordinary observations, without source bounds or safety facts. -/
structure TensorMetadata where
  (ordinary fp32 strided cuda negative conjugate contiguous : Bool)
  device : Nat
  shape : List Nat
  strides : List Int
  numel : Nat

def commonChecks (device : Nat) (t : TensorMetadata) := decide (
  t.ordinary=Bool.true ∧ t.fp32=Bool.true ∧ t.strided=Bool.true ∧ t.cuda=Bool.true ∧
  t.negative=Bool.false ∧ t.conjugate=Bool.false ∧ t.device=device)

def inputChecks (g : Geometry) (q k v qs ks : TensorMetadata) :=
  [q,k,v,qs,ks].all (commonChecks q.device) && decide (
    q.shape=[g.batches,g.heads,g.context,g.headDim] ∧ k.shape=q.shape ∧ v.shape=q.shape ∧
    q.strides=(g.stride .query).raw ∧ k.strides=(g.stride .key).raw ∧ v.strides=(g.stride .value).raw)

def launchChecks (g : Geometry) (qs ks : TensorMetadata) := decide (
  qs.contiguous=Bool.true ∧ ks.contiguous=Bool.true ∧
  span g .queryScale≤qs.numel ∧ span g .keyScale≤ks.numel ∧
  countQ g≤2147483647 ∧ g.batches*g.heads≤65535 ∧ HostShape.blockD g.headDim≤1048576 ∧
  envelope g<2^63)

def pointerChecks (g : Geometry) (base : Role→Nat) :=
  roles.all fun r=>decide (base r%width r=0 ∧ base r+width r*envelope g<2^64)

theorem launchable (g : Geometry) (qs ks : TensorMetadata) (h : launchChecks g qs ks=Bool.true) :
    NativeLaunch.CudaLaunchable ⟨g.batches,g.heads,g.context⟩ (HostShape.blockD g.headDim) := by
  simp only [launchChecks,decide_eq_true_eq] at h
  exact ⟨h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2.1⟩

theorem formed_signed (g : Geometry) (qs ks : TensorMetadata) (h : launchChecks g qs ks=Bool.true)
    (n : Nat) (hn : n<envelope g) : signed64 n=(n : Int) ∧ wrap64 n=n := by
  simp only [launchChecks,decide_eq_true_eq] at h
  have hb : n<2^63 := lt_trans hn h.2.2.2.2.2.2.2
  exact ⟨signed64_exact hb,wrap64_exact hb⟩

theorem formed_pointer (g : Geometry) (base : Role→Nat) (h : pointerChecks g base=Bool.true)
    (r : Role) (n : Nat) (hn : n<envelope g) :
    base r%width r=0 ∧ base r+width r*n+width r<2^64 ∧
      Verified.NativeOffsets.bytePointer (base r) (width r) n=base r+width r*n := by
  simp only [pointerChecks,List.all_eq_true,decide_eq_true_eq] at h
  have member : r∈roles := by cases r <;> simp [roles]
  have hb:=h r member
  have hm:=Nat.mul_le_mul_left (width r) (show n+1≤envelope g by omega)
  simp only [Nat.mul_add,Nat.mul_one] at hm
  have he : base r+width r*n+width r<2^64 := by omega
  refine ⟨hb.1,he,?_⟩
  rw [Verified.NativeOffsets.bytePointer_mod,Nat.mul_comm,wrap64,Nat.mod_eq_of_lt (show base r+width r*n<2^64 by omega)]

theorem active_bound (g : Geometry) (r : Role) (hzs : r≠Role.queryScale) (hks : r≠Role.keyScale)
    (z h n d : Nat) (hz : z<g.batches) (hh : h<g.heads) (hn : n<g.context) (hd : d<g.headDim) :
    (g.stride r).at z h n d<span g r := by
  have hb:=Nat.mul_le_mul_right (g.stride r).batch (show z≤g.batches-1 by omega)
  have hh':=Nat.mul_le_mul_right (g.stride r).head (show h≤g.heads-1 by omega)
  have hn':=Nat.mul_le_mul_right (g.stride r).row (show n≤g.context-1 by omega)
  have hd':=Nat.mul_le_mul_right (g.stride r).dim (show d≤g.headDim-1 by omega)
  have nz : ¬(g.batches=0 ∨ g.heads=0 ∨ g.context=0 ∨ g.headDim=0) := by omega
  cases r <;> try contradiction
  all_goals simp only [span,if_neg nz,Strides.at] at * <;> omega

theorem extent_bound (g : Geometry) (r : Role) (z h n d : Nat)
    (hz : z≤g.batches) (hh : h≤g.heads) (hn : n≤g.context+256) (hd : d≤HostShape.blockD g.headDim) :
    (g.stride r).at z h n d<extent g r := by
  have hb:=Nat.mul_le_mul_right (g.stride r).batch hz
  have hh':=Nat.mul_le_mul_right (g.stride r).head hh
  have hn':=Nat.mul_le_mul_right (g.stride r).row hn
  have hd':=Nat.mul_le_mul_right (g.stride r).dim hd
  simp only [Strides.at,extent]
  omega

theorem matrix_formed (g : Geometry) (r : Role) (hqs : r≠Role.queryScale) (hks : r≠Role.keyScale)
    (z h n d : Nat) (hz : z≤g.batches) (hh : h≤g.heads) (hn : n≤g.context+256) (hd : d≤HostShape.blockD g.headDim) :
    (g.stride r).at z h n d<envelope g := by
  have hb:=extent_bound g r z h n d hz hh hn hd
  cases r <;> try contradiction
  all_goals simp only [envelope] at * <;> omega

theorem packed_formed (g : Geometry) (r : Role) (hrole : r=Role.queryScale ∨ r=Role.keyScale)
    (head block : Nat) (hh : head≤g.batches*g.heads)
    (hb : block≤(if r=Role.queryScale then countQ g+1 else countK g+1)) :
    head*(if r=Role.queryScale then countQ g else countK g)+block<envelope g := by
  rcases hrole with rfl|rfl
  · simp only [↓reduceIte] at *
    have hm:=Nat.mul_le_mul_right (countQ g) hh
    unfold envelope
    omega
  · simp only [show Role.keyScale≠Role.queryScale by decide,↓reduceIte] at *
    have hm:=Nat.mul_le_mul_right (countK g) hh
    unfold envelope
    omega

/-- Covers all padded lanes and the unused final pointer advancement in either
source helper. No floating-point premise appears in these bounds. -/
theorem padded_query (g : Geometry) (pid lane : Nat) (hp : pid*128<g.context) (hl : lane<128) :
    pid*128+lane≤g.context+256 := by omega

theorem padded_key (g : Geometry) (start lane : Nat) (hs : start<g.context) (hl : lane<64) :
    start+lane≤g.context+256 ∧ start+64+lane≤g.context+256 := by omega

#axiomsClean launchable
#axiomsClean formed_signed
#axiomsClean formed_pointer
#axiomsClean active_bound
#axiomsClean matrix_formed
#axiomsClean packed_formed
#axiomsClean padded_query
#axiomsClean padded_key
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
