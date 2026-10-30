import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Host
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 4000000

/-- Exact logical capacities of the supported FP32 storage profile. Head gaps
are retained; neither source nor model inserts a contiguous copy. -/
def extent (c : Shape) : Region → Nat
  | .q | .k | .w => c.Z*c.keyPitch
  | .v => c.Z*c.valuePitch
  | .u => c.H*c.K
  | .h0 | .ht => c.Z*c.K*c.V
  | .o => c.NK*c.Z*c.T*c.V
  | .result => c.Z*c.T*c.V
  | .outside => 0

def signed32 (x : Int) : Prop := -(2^31) ≤ x ∧ x<2^31

/-- All setup arithmetic, including masked tail lanes and the empty reverse
start, is checked before pointer widening. Temporal pointer updates use i64. -/
def setupExpressions (c : Shape) (p : Program) (j l : Nat) : List Int :=
  let z : Int := p.z
  let ik : Int := p.ik
  let iv : Int := p.iv
  let bk : Int := c.BK
  let bv : Int := c.BV
  let e := ik*bk+j
  let f := iv*bv+l
  let startK := if c.reverse then ((c.T:Int)-1)*c.K else 0
  let startV := if c.reverse then ((c.T:Int)-1)*c.V else 0
  [z,ik,iv,bk,bv,p.z%c.H,ik*bk,iv*bv,e,f,
   z*c.keyPitch,z*c.valuePitch,z*c.keyPitch+e,z*c.valuePitch+f,
   startK,startV,z*c.keyPitch+e+startK,z*c.valuePitch+f+startV,
   ik*c.B,ik*c.B*c.H,z+ik*c.B*c.H,(c.T:Int)*c.V,
   (z+ik*c.B*c.H)*(c.T*c.V),
   (z+ik*c.B*c.H)*(c.T*c.V)+f,
   (z+ik*c.B*c.H)*(c.T*c.V)+f+startV,
   ((p.z%c.H:Nat):Int)*c.K+j, ((p.z%c.H:Nat):Int)*c.K+j+ik*bk,
   z*c.K,z*c.K*c.V,e*c.V,z*c.K*c.V+e*c.V,z*c.K*c.V+e*c.V+f]

/-- Source pointer at each visit, including the harmless final increment. -/
def temporalPointer (c : Shape) (base dimension n : Int) : Int :=
  base + (if c.reverse then ((c.T:Int)-1)*dimension else 0) +
    n*(if c.reverse then -dimension else dimension)

structure Domain (c : Shape) : Prop where
  sizes : Sizes c
  keyPitch : c.T*c.K≤c.keyPitch
  valuePitch : c.T*c.V≤c.valuePitch
  bkSource : c.BK=min (2^Nat.clog 2 c.K) 32
  bvSource : c.BV=min (2^Nat.clog 2 c.V) 32
  gridX : c.NV≤2^31-1
  gridY : c.NK≤65535
  gridZ : c.Z≤65535
  i32 : ∀ p, Valid c p → ∀ j<c.BK, ∀ l<c.BV, ∀ x∈setupExpressions c p j l, signed32 x
  capacities : ∀ r, 4*extent c r<2^63

/-- The original affine pointer advances equal logical forward/reverse visits. -/
theorem pointer_visit (c : Shape) (base d : Int) (n : Nat) (hn : n<c.T) :
    temporalPointer c base d n=base+(visit c n:Int)*d := by
  have hcast : ((c.T-1-n:Nat):Int)=(c.T:Int)-1-n := by omega
  cases hr : c.reverse <;> simp only [temporalPointer,visit,hr,Bool.false_eq_true,if_false,if_true]
  · ring
  · rw [hcast]
    ring

theorem key_bound (c : Shape) (hc : Domain c) (z t e : Nat)
    (hz : z<c.Z) (ht : t<c.T) (he : e<c.K) : inputKey c z t e<extent c .k := by
  have localBound : t*c.K+e<c.keyPitch := by nlinarith [hc.keyPitch]
  unfold inputKey extent
  nlinarith

theorem value_bound (c : Shape) (hc : Domain c) (z t f : Nat)
    (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) : inputValue c z t f<extent c .v := by
  have localBound : t*c.V+f<c.valuePitch := by nlinarith [hc.valuePitch]
  unfold inputValue extent
  nlinarith

theorem bias_bound (c : Shape) (hc : Sizes c) (z e : Nat) (he : e<c.K) :
    biasAddress c z e<extent c .u := by
  have hz := Nat.mod_lt z hc.H
  unfold biasAddress extent
  nlinarith

theorem state_address_bound (c : Shape) (z e f : Nat)
    (hz : z<c.Z) (he : e<c.K) (hf : f<c.V) : stateAddress c z e f<extent c .h0 := by
  have h : z*c.K+e<c.Z*c.K := by nlinarith
  unfold stateAddress extent
  nlinarith

theorem output_capacity (c : Shape) {r : Region} {b : Nat} (h : WholeOutput c r b) : b<extent c r := by
  rcases h with (⟨rfl,h⟩|⟨_,rfl,h⟩)|⟨rfl,h⟩ <;> exact h

/-- Complete active read footprint. U and H0 still occur when T=0. -/
def Reads (c : Shape) (p : Program) (r : Region) (b : Nat) : Prop :=
  (∃ j<c.BK, keyIndex c p j<c.K ∧ r=.u ∧ b=biasAddress c p.z (keyIndex c p j)) ∨
  (c.initial ∧ ∃ j<c.BK, ∃ l<c.BV, keyIndex c p j<c.K ∧ valueIndex c p l<c.V ∧
    r=.h0 ∧ b=stateAddress c p.z (keyIndex c p j) (valueIndex c p l)) ∨
  (∃ n<c.T, ∃ j<c.BK, keyIndex c p j<c.K ∧ (r=.q ∨ r=.k ∨ r=.w) ∧
    b=inputKey c p.z (visit c n) (keyIndex c p j)) ∨
  (∃ n<c.T, ∃ l<c.BV, valueIndex c p l<c.V ∧ r=.v ∧
    b=inputValue c p.z (visit c n) (valueIndex c p l))

theorem reads_bounded (c : Shape) (hc : Domain c) (p : Program) (hp : Valid c p)
    {r : Region} {b : Nat} (h : Reads c p r b) : b<extent c r := by
  rcases h with ⟨j,_,hj,rfl,rfl⟩ | ⟨_,j,_,l,_,he,hf,rfl,rfl⟩ |
    ⟨n,hn,j,_,he,hr,rfl⟩ | ⟨n,hn,l,_,hf,rfl,rfl⟩
  · exact bias_bound c hc.sizes p.z _ hj
  · exact state_address_bound c p.z _ _ hp.1 he hf
  · rcases hr with rfl|rfl|rfl <;> exact key_bound c hc p.z _ _ hp.1 (visit_lt c hn) he
  · exact value_bound c hc p.z _ _ hp.1 (visit_lt c hn) hf

/-- A single normal allocation element has four in-bounds byte addresses. -/
theorem byte_bound (c : Shape) (hc : Domain c) (r : Region) (b j : Nat)
    (hb : b<extent c r) (hj : j<4) : 4*b+j<4*extent c r ∧ 4*b+j<2^63 := by
  have he := hc.capacities r
  constructor <;> omega

#axiomsClean reads_bounded
#axiomsClean pointer_visit
#axiomsClean byte_bound
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
