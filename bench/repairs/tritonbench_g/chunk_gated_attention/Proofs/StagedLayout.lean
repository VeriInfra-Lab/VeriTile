import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RebasedFlat
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.HostStaging

/-! Host-produced padded dimensions discharge tile activity, every allocation
bound, and within-program collision conditions. No new native-safe shape bound
or kernel output equation is a premise. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.StagedLayout
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

structure Canonical (p : Layout) : Prop where
  keyHead : p.keyHead = p.T*p.K
  keyTime : p.keyTime = p.K
  keyFeature : p.keyFeature = 1
  valueHead : p.valueHead = p.T*p.V
  valueTime : p.valueTime = p.V
  valueFeature : p.valueFeature = 1
  historyHead : p.historyHead = p.NT*p.K*p.V
  historyKey : p.historyKey = p.V
  historyValue : p.historyValue = 1
  time : p.T = p.NT*p.BT
  key : p.K/p.BK*p.BK = p.K
  value : p.V/p.BV*p.BV = p.V
  bt : 0 < p.BT
  bk : 0 < p.BK
  bv : 0 < p.BV

structure TileValid (p : Layout) (heads : Nat) (l : Launch) : Prop where
  head : l.head < heads
  key : l.key < p.K/p.BK
  value : l.value < p.V/p.BV

theorem decoded_valid (p : Layout) (heads base : Nat) (u : BlockState)
    (h : u.pids 0+base < heads*(p.K/p.BK)*(p.V/p.BV)) :
    TileValid p heads (RebasedInit.decoded p base u) := by
  have hx : 0 < p.V/p.BV := by
    by_contra hn
    have hz : p.V/p.BV = 0 := Nat.eq_zero_of_not_pos hn
    simp [hz] at h
  have hy : 0 < p.K/p.BK := by
    by_contra hn
    have hz : p.K/p.BK = 0 := Nat.eq_zero_of_not_pos hn
    simp [hz] at h
  have hd := Grid.decode_bounded hx hy h
  exact ⟨hd.2.2,hd.2.1,hd.1⟩

theorem key_full (p : Layout) (heads : Nat) (l : Launch) (hp : Canonical p) (hl : TileValid p heads l) :
    l.key*p.BK+p.BK ≤ p.K := by
  have h := Nat.mul_le_mul_right p.BK (Nat.succ_le_iff.mpr hl.key)
  simpa [Nat.succ_mul,hp.key] using h

theorem value_full (p : Layout) (heads : Nat) (l : Launch) (hp : Canonical p) (hl : TileValid p heads l) :
    l.value*p.BV+p.BV ≤ p.V := by
  have h := Nat.mul_le_mul_right p.BV (Nat.succ_le_iff.mpr hl.value)
  simpa [Nat.succ_mul,hp.value] using h

theorem full_tile (p : Layout) (heads : Nat) (l : Launch) (hp : Canonical p) (hl : TileValid p heads l)
    (c : Nat) (hc : c < p.NT) : RebasedAccess.Full p l c := by
  refine ⟨key_full p heads l hp hl,value_full p heads l hp hl,?_,hp.bt,?_,?_⟩
  · simpa [hp.time,Nat.succ_mul] using Nat.mul_le_mul_right p.BT (Nat.succ_le_iff.mpr hc)
  · simp [hp.keyTime,hp.keyFeature]
  · simp [hp.valueTime,hp.valueFeature]

theorem key_index (p : Layout) (heads : Nat) (l : Launch) (hp : Canonical p) (hl : TileValid p heads l)
    (e : Fin p.BK) : keyIndex p l e.val < p.K := by
  have := key_full p heads l hp hl
  unfold keyIndex
  omega

theorem value_index (p : Layout) (heads : Nat) (l : Launch) (hp : Canonical p) (hl : TileValid p heads l)
    (j : Fin p.BV) : valueIndex p l j.val < p.V := by
  have := value_full p heads l hp hl
  unfold valueIndex
  omega

theorem time_index (p : Layout) (hp : Canonical p) (c : Nat) (hc : c < p.NT)
    (t : Fin p.BT) : timeIndex p c t.val < p.T := by
  rw [hp.time]
  exact Address.flatten2_bounded hc t.isLt

structure Allocated (p : Layout) (heads : Nat) (r : Regions) (gateK hasSeed storeFinal : Bool)
    (bounds : RegionBounds) : Prop where
  key : heads*p.T*p.K ≤ bounds r.key
  value : heads*p.T*p.V ≤ bounds r.value
  gate : heads*p.T*(if gateK then p.K else p.V) ≤ bounds r.gate
  history : heads*p.NT*p.K*p.V ≤ bounds r.history
  seed : hasSeed = Bool.true → heads*p.K*p.V ≤ bounds r.seed
  final : storeFinal = Bool.true → heads*p.K*p.V ≤ bounds r.final

theorem key_bounds (p : Layout) (heads : Nat) (l : Launch) (r : RegionName)
    (hp : Canonical p) (hl : TileValid p heads l) (c : Nat) (hc : c < p.NT)
    (bounds : RegionBounds) (hb : heads*p.T*p.K ≤ bounds r) :
    RebasedAccessSafety.KeyBounds p l r c bounds := by
  intro e t _
  have h := lt_of_lt_of_le (Address.flatten3_bounded hl.head (time_index p hp c hc t)
    (key_index p heads l hp hl e)) hb
  convert h using 1 <;> simp only [hp.keyHead,hp.keyFeature,hp.keyTime,Nat.mul_one,Nat.zero_add,timeIndex,keyIndex] <;> ring

theorem value_bounds (p : Layout) (heads : Nat) (l : Launch) (r : RegionName)
    (hp : Canonical p) (hl : TileValid p heads l) (c : Nat) (hc : c < p.NT)
    (bounds : RegionBounds) (hb : heads*p.T*p.V ≤ bounds r) :
    RebasedAccessSafety.ValueBounds p l r c bounds := by
  intro t j _
  have h := lt_of_lt_of_le (Address.flatten3_bounded hl.head (time_index p hp c hc t)
    (value_index p heads l hp hl j)) hb
  convert h using 1 <;> simp only [hp.valueHead,hp.valueFeature,hp.valueTime,Nat.mul_one,Nat.zero_add,timeIndex,valueIndex] <;> ring

theorem state_bounds (p : Layout) (heads : Nat) (l : Launch) (r : RegionName)
    (hp : Canonical p) (hl : TileValid p heads l)
    (bounds : RegionBounds) (hb : heads*p.K*p.V ≤ bounds r) :
    RebasedAccessSafety.StateBounds p l r bounds := by
  intro e j _
  have h := lt_of_lt_of_le (Address.flatten3_bounded hl.head (key_index p heads l hp hl e)
    (value_index p heads l hp hl j)) hb
  convert h using 1 <;> simp only [Nat.mul_one,Nat.zero_add,keyIndex,valueIndex] <;> ring

theorem history_bounds (p : Layout) (heads : Nat) (l : Launch) (r : RegionName)
    (hp : Canonical p) (hl : TileValid p heads l) (c : Nat) (hc : c < p.NT)
    (bounds : RegionBounds) (hb : heads*p.NT*p.K*p.V ≤ bounds r) :
    RebasedAccessSafety.HistoryBounds p l r c bounds := by
  intro e j _
  have h := lt_of_lt_of_le (Address.history_bounded hl.head hc (key_index p heads l hp hl e)
    (value_index p heads l hp hl j)) hb
  convert h using 1 <;> simp only [hp.historyHead,hp.historyKey,hp.historyValue,Nat.mul_one,Nat.zero_add,keyIndex,valueIndex] <;> ring

theorem endpoint_time (p : Layout) (hp : Canonical p) (c : Nat) (hc : c < p.NT) :
    c*p.BT+p.BT-1 < p.T := by
  have h := Nat.mul_le_mul_right p.BT (Nat.succ_le_iff.mpr hc)
  simp only [Nat.succ_mul] at h
  have := hp.bt
  rw [hp.time]
  omega

theorem key_endpoint_bounds (p : Layout) (heads : Nat) (l : Launch) (r : RegionName)
    (hp : Canonical p) (hl : TileValid p heads l) (c : Nat) (hc : c < p.NT)
    (bounds : RegionBounds) (hb : heads*p.T*p.K ≤ bounds r) :
    RebasedAccessSafety.EndpointBounds p l r c p.keyHead p.keyTime p.keyFeature p.BK l.key bounds := by
  intro e _
  have h := lt_of_lt_of_le (Address.flatten3_bounded hl.head (endpoint_time p hp c hc)
    (key_index p heads l hp hl e)) hb
  convert h using 1 <;> simp only [hp.keyHead,hp.keyTime,hp.keyFeature,Nat.mul_one,Nat.zero_add,keyIndex] <;> ring

theorem value_endpoint_bounds (p : Layout) (heads : Nat) (l : Launch) (r : RegionName)
    (hp : Canonical p) (hl : TileValid p heads l) (c : Nat) (hc : c < p.NT)
    (bounds : RegionBounds) (hb : heads*p.T*p.V ≤ bounds r) :
    RebasedAccessSafety.EndpointBounds p l r c p.valueHead p.valueTime p.valueFeature p.BV l.value bounds := by
  intro j _
  have h := lt_of_lt_of_le (Address.flatten3_bounded hl.head (endpoint_time p hp c hc)
    (value_index p heads l hp hl j)) hb
  convert h using 1 <;> simp only [hp.valueHead,hp.valueTime,hp.valueFeature,Nat.mul_one,Nat.zero_add,valueIndex] <;> ring

theorem bounds (p : Layout) (heads : Nat) (l : Launch) (r : Regions)
    (gateK hasSeed storeFinal : Bool) (hp : Canonical p) (hl : TileValid p heads l)
    (b : RegionBounds) (hb : Allocated p heads r gateK hasSeed storeFinal b) :
    RebasedSafety.Bounds p l r gateK hasSeed storeFinal b := by
  refine ⟨fun h => state_bounds p heads l r.seed hp hl b (hb.seed h),?_,
    fun h => state_bounds p heads l r.final hp hl b (hb.final h)⟩
  intro c hc
  refine ⟨key_bounds p heads l r.key hp hl c hc b hb.key,
    value_bounds p heads l r.value hp hl c hc b hb.value,
    history_bounds p heads l r.history hp hl c hc b hb.history,?_⟩
  constructor
  · intro h; subst gateK
    exact key_bounds p heads l r.gate hp hl c hc b hb.gate
  · intro h; subst gateK
    exact key_endpoint_bounds p heads l r.gate hp hl c hc b hb.gate
  · intro h; subst gateK
    exact value_bounds p heads l r.gate hp hl c hc b hb.gate
  · intro h; subst gateK
    exact value_endpoint_bounds p heads l r.gate hp hl c hc b hb.gate

/-- Row-major coordinate equality, used for history/state ownership. -/
theorem split_address {n a b x y : Nat} (hx : x < n) (hy : y < n)
    (he : a*n+x = b*n+y) : a=b ∧ x=y := by
  have hn : 0 < n := Nat.zero_lt_of_lt hx
  have hd := congrArg (fun z => z/n) he
  have hm := congrArg (fun z => z%n) he
  simp [Nat.add_comm _ x,Nat.add_comm _ y,Nat.add_mul_div_right _ _ hn,
    Nat.div_eq_of_lt hx,Nat.div_eq_of_lt hy,Nat.mod_eq_of_lt hx,Nat.mod_eq_of_lt hy] at hd hm
  exact ⟨hd,hm⟩

theorem state_noCollision (p : Layout) (l : Launch) :
    RealHStore.NoCollision p l (RealHFinal.base p l) p.V 1 := by
  intro e f i j he hf heq
  have h := split_address he.2 hf.2 (show
      (l.head*p.K+keyIndex p l e.val)*p.V+valueIndex p l i.val =
      (l.head*p.K+keyIndex p l f.val)*p.V+valueIndex p l j.val by
    simpa only [RealHStore.address,RealHFinal.base,keyIndex,valueIndex,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using heq)
  have he' : e.val = f.val := by unfold keyIndex at h; omega
  have hj' : i.val = j.val := by unfold valueIndex at h; omega
  exact ⟨Fin.ext he',Fin.ext hj'⟩

theorem history_noCollision (p : Layout) (l : Launch) (hp : Canonical p) :
    RealHLoop.HistoryNoCollision p l := by
  intro c d e f i j hc hd he hf heq
  have h := split_address he.2 hf.2 (show
      ((l.head*p.NT+c)*p.K+keyIndex p l e.val)*p.V+valueIndex p l i.val =
      ((l.head*p.NT+d)*p.K+keyIndex p l f.val)*p.V+valueIndex p l j.val by
    simpa only [RealHLoop.address,RealHStore.address,RealHStep.historyBase,hp.historyHead,
      hp.historyKey,hp.historyValue,keyIndex,valueIndex,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using heq)
  have h' := split_address he.1 hf.1 h.1
  have hcd : c=d := by omega
  have hef : e.val=f.val := by unfold keyIndex at h'; omega
  have hij : i.val=j.val := by unfold valueIndex at h; omega
  exact ⟨hcd,Fin.ext hef,Fin.ext hij⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.StagedLayout
