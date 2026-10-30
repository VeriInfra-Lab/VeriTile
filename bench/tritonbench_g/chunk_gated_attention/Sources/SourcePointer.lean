import bench.tritonbench_g.chunk_gated_attention.Sources.RequestCore

/-! Shared source pointer polynomials and local finite-word interpretation. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.SourceAddress
open VeriTile.Triton HostFP Address
noncomputable section
set_option maxHeartbeats 3000000

structure Access (a : Allocation.Buffer) where
  offset : Nat
  sourceWord : BitVec 64
  wordEquation : sourceWord=word offset
  bounded : offset<a.extent

theorem Access.signed {a : Allocation.Buffer} (v : Access a) : v.sourceWord.toInt=(v.offset : Int) := by
  rw [v.wordEquation]
  exact word_exact (lt_of_lt_of_le v.bounded a.extent_bound)

theorem Access.address {a : Allocation.Buffer} (v : Access a) :
    a.nativeAddress v.sourceWord=a.address v.offset := by
  rw [v.wordEquation]
  exact a.native_address v.offset v.bounded

theorem Access.read {α : Type} {a : Allocation.Buffer} (v : Access a) (m : Allocation.Cells α a) :
    m.physical (a.nativeAddress v.sourceWord)=m.typed v.offset := by
  rw [v.address]
  exact m.coordinate v.offset v.bounded

theorem padded_cover (n b : Nat) (hb : 0<b) : n≤padded n b := by
  have hm := Nat.mod_lt (n+b-1) hb
  have he := Nat.mod_add_div (n+b-1) b
  rw [Nat.mul_comm b] at he
  unfold padded chunks
  omega

theorem padded_tile {n b tile lane : Nat} (hb : 0<b)
    (ht : tile<padded n b/b) (hl : lane<b) : tile*b+lane<padded n b := by
  have he : padded n b/b*b=padded n b := by simp [padded,hb.ne']
  have hm := Nat.mul_le_mul_right b (Nat.succ_le_iff.mpr ht)
  simp only [Nat.succ_mul,he] at hm
  omega

structure InnerCoord (c : HostFP.Inner) where
  head : Fin c.heads
  keyTile : Fin (c.PK/c.BK)
  valueTile : Fin (c.PV/c.BV)
  chunk : Fin c.NT
  time : Fin c.BT
  key : Fin c.BK
  value : Fin c.BV

def InnerCoord.k {c : HostFP.Inner} (i : InnerCoord c) := i.keyTile.val*c.BK+i.key.val
def InnerCoord.v {c : HostFP.Inner} (i : InnerCoord c) := i.valueTile.val*c.BV+i.value.val
def InnerCoord.t {c : HostFP.Inner} (i : InnerCoord c) := i.chunk.val*c.BT+i.time.val
def InnerCoord.last {c : HostFP.Inner} (i : InnerCoord c) := i.chunk.val*c.BT+c.BT-1

theorem inner_bounds (c : HostFP.Inner) (i : InnerCoord c) (ht : 0<c.BT) (hk : 0<c.BK) (hv : 0<c.BV) :
    i.k<c.PK ∧ i.v<c.PV ∧ i.t<c.PT ∧ i.last<c.PT := by
  have hK := padded_tile (n := c.K) hk i.keyTile.isLt i.key.isLt
  have hV := padded_tile (n := c.V) hv i.valueTile.isLt i.value.isLt
  have hT : i.t<c.PT := by
    exact Address.flatten2_bounded i.chunk.isLt i.time.isLt
  have hL : i.last<c.PT := by
    have hm := Nat.mul_le_mul_right c.BT (Nat.succ_le_iff.mpr i.chunk.isLt)
    change (i.chunk.val+1)*c.BT≤c.PT at hm
    dsimp [InnerCoord.last]
    simp only [Nat.add_mul,Nat.one_mul] at hm
    omega
  exact ⟨hK,hV,hT,hL⟩

inductive InnerSite where
  | seed | key | value | history | gateKey | gateKeyEnd | gateValue | gateValueEnd | final
  deriving DecidableEq

def region (r : HostRequest.Regions) : InnerSite → RegionName
  | .seed => r.seed | .key => r.key | .value => r.value | .history => r.history
  | .gateKey | .gateKeyEnd | .gateValue | .gateValueEnd => r.gate
  | .final => r.final

def enabled (c : HostFP.Inner) : InnerSite → Prop
  | .seed => c.hasSeed=Bool.true
  | .final => c.storeFinal=Bool.true
  | .gateKey | .gateKeyEnd => c.gateK=Bool.true
  | .gateValue | .gateValueEnd => c.gateK=Bool.false
  | _ => True

/-- Exact canonical base plus local-lane expressions from Python's nine
recurrent pointer occurrences. Every block-pointer offset itself is zero. -/
def innerWord (c : HostFP.Inner) (i : InnerCoord c) : InnerSite → BitVec 64
  | .seed | .final => word i.head.val*word c.PK*word c.PV +
      word i.keyTile.val*word c.BK*word c.PV + word i.valueTile.val*word c.BV +
      word i.key.val*word c.PV + word i.value.val
  | .key | .gateKey => word i.head.val*word (c.PT*c.PK) +
      word i.keyTile.val*word c.BK*word 1 + word i.chunk.val*word c.BT*word c.PK +
      word i.key.val*word 1 + word i.time.val*word c.PK
  | .value | .gateValue => word i.head.val*word (c.PT*c.PV) +
      word i.chunk.val*word c.BT*word c.PV + word i.valueTile.val*word c.BV*word 1 +
      word i.time.val*word c.PV + word i.value.val*word 1
  | .history => word i.head.val*word (c.NT*c.PK*c.PV) +
      word i.chunk.val*word c.PK*word c.PV + word i.keyTile.val*word c.BK*word c.PV +
      word i.valueTile.val*word c.BV*word 1 + word i.key.val*word c.PV + word i.value.val*word 1
  | .gateKeyEnd => word i.head.val*word (c.PT*c.PK) +
      (word i.chunk.val*word c.BT+word c.BT-word 1)*word c.PK +
      word i.keyTile.val*word c.BK*word 1 + word i.key.val*word 1
  | .gateValueEnd => word i.head.val*word (c.PT*c.PV) +
      (word i.chunk.val*word c.BT+word c.BT-word 1)*word c.PV +
      word i.valueTile.val*word c.BV*word 1 + word i.value.val*word 1

def innerOffset (c : HostFP.Inner) (i : InnerCoord c) : InnerSite → Nat
  | .seed | .final => i.head.val*(c.PK*c.PV)+i.k*c.PV+i.v
  | .key | .gateKey => i.head.val*(c.PT*c.PK)+i.t*c.PK+i.k
  | .value | .gateValue => i.head.val*(c.PT*c.PV)+i.t*c.PV+i.v
  | .history => i.head.val*(c.NT*c.PK*c.PV)+i.chunk.val*c.PK*c.PV+i.k*c.PV+i.v
  | .gateKeyEnd => i.head.val*(c.PT*c.PK)+i.last*c.PK+i.k
  | .gateValueEnd => i.head.val*(c.PT*c.PV)+i.last*c.PV+i.v

theorem endpoint_word (chunk BT : Nat) (h : 0<BT) :
    word chunk*word BT+word BT-word 1=word (chunk*BT+BT-1) := by
  simp only [word,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact BitVec.ofNat_sub_ofNat_of_le _ _ (by norm_num) (by omega)

theorem inner_word (c : HostFP.Inner) (i : InnerCoord c) (site : InnerSite) (h : 0<c.BT) :
    innerWord c i site=word (innerOffset c i site) := by
  cases site <;> simp only [innerWord,innerOffset]
  all_goals try rw [endpoint_word _ _ h]
  all_goals simp only [InnerCoord.k,InnerCoord.v,InnerCoord.t,InnerCoord.last,word,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  all_goals congr 1 <;> ring


end
end VeriTile.Bench.Repairs.ChunkGatedAttention.SourceAddress
