import bench.tritonbench_g.iv_dependent_matmul.Sources.Mathematical_Host

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
open VeriTile.Triton FP.Structural FPValue Kernels
noncomputable section
set_option maxHeartbeats 3000000

def envelopeA (r : Request) := r.rows*max r.inner 1+(count r.parameters+3)*r.blockInner
def envelopeB (r : Request) := ((count r.parameters+3)*r.blockInner)*max r.columns 1+r.columns
def envelopeC (r : Request) := (tileRows r*r.blockRows)*max r.columns 1+tileColumns r*r.blockColumns

/-- Sufficient finite-index/byte-pointer domain, including initial prefetches
and the final `k+3` pointer formation of the three-iteration schedule.
It is separate from the unrestricted natural-index execution theorem. -/
structure IndexDomain (r : Request) (s : State Word) (a : Allocation r s) : Prop where
  rows : r.rows+r.blockRows<2^31
  columns : r.columns+r.blockColumns<2^31
  inner : r.inner+r.blockInner<2^31
  grid : programs r<2^31
  counter : count r.parameters+3<2^31
  addressA : envelopeA r<2^31
  addressB : envelopeB r<2^31
  addressC : envelopeC r<2^31
  byteA : a.baseA+4*envelopeA r<2^64
  byteB : a.baseB+4*envelopeB r<2^64
  byteC : a.baseC+4*envelopeC r<2^64

theorem nonempty_program (r : Request) (pid : Nat) (hp : pid<programs r) :
    0<r.rows ∧ 0<r.columns := by
  have hb := r.positiveRowsBlock
  have hc := r.positiveColumnsBlock
  constructor
  · by_contra hn
    have hz : r.rows=0 := by omega
    have hdiv : (r.blockRows-1)/r.blockRows=0 := Nat.div_eq_of_lt (by omega)
    simp [programs,tileRows,hz,hdiv] at hp
  · by_contra hn
    have hz : r.columns=0 := by omega
    have hdiv : (r.blockColumns-1)/r.blockColumns=0 := Nat.div_eq_of_lt (by omega)
    simp [programs,tileColumns,hz,hdiv] at hp

theorem source_subtraction (r : Request) (k : Nat) (hk : k<count r.parameters) :
    k*r.blockInner<r.inner ∧ (r.inner : Int)-(k*r.blockInner : Nat)=(r.inner-k*r.blockInner : Nat) := by
  have hb := r.positiveInnerBlock
  have hd := Nat.div_mul_le_self (r.inner+r.blockInner-1) r.blockInner
  have hm := Nat.mul_le_mul_right r.blockInner (Nat.succ_le_of_lt hk)
  simp only [count,Request.parameters] at hm
  rw [Nat.succ_mul] at hm
  have hsub : r.inner+r.blockInner-1<r.inner+r.blockInner := by omega
  have hlt : k*r.blockInner<r.inner := by omega
  exact ⟨hlt,by omega⟩

theorem a_offset (r : Request) (s : State Word) (k : Nat) (i : TileIndex [r.blockRows,r.blockInner]) :
    aOffset r.parameters s k i=(row r.parameters s i.1%r.rows)*max r.inner 1+(k*r.blockInner+i.2.1.val) := by
  simp only [aOffset,aBase,Request.parameters,Nat.mul_one]
  ring

theorem b_offset (r : Request) (s : State Word) (k : Nat) (i : TileIndex [r.blockInner,r.blockColumns]) :
    bOffset r.parameters s k i=(k*r.blockInner+i.1.val)*max r.columns 1+(col r.parameters s i.2.1%r.columns) := by
  simp only [bOffset,bBase,Request.parameters,Nat.mul_one]
  ring

theorem loadA_bound (r : Request) (s : State Word) (pid k : Nat) (hp : pid<programs r)
    (i : TileIndex [r.blockRows,r.blockInner]) (hm : i.2.1.val<r.inner-k*r.blockInner) :
    aOffset r.parameters (instanceState pid s) k i<r.rows*r.inner := by
  have hn := (nonempty_program r pid hp).1
  have hr := Nat.mod_lt (row r.parameters (instanceState pid s) i.1) hn
  have hk : k*r.blockInner+i.2.1.val<r.inner := by omega
  have hs : max r.inner 1=r.inner := by omega
  rw [a_offset,hs]
  have hh := Nat.mul_le_mul_right r.inner (Nat.succ_le_of_lt hr)
  nlinarith

theorem loadB_bound (r : Request) (s : State Word) (pid k : Nat) (hp : pid<programs r)
    (i : TileIndex [r.blockInner,r.blockColumns]) (hm : i.1.val<r.inner-k*r.blockInner) :
    bOffset r.parameters (instanceState pid s) k i<r.inner*r.columns := by
  have hn := (nonempty_program r pid hp).2
  have hc := Nat.mod_lt (col r.parameters (instanceState pid s) i.2.1) hn
  have hk : k*r.blockInner+i.1.val<r.inner := by omega
  have hs : max r.columns 1=r.columns := by omega
  rw [b_offset,hs]
  have hh := Nat.mul_le_mul_right r.columns (Nat.succ_le_of_lt hk)
  nlinarith

theorem store_bound (r : Request) (s : State Word) (pid : Nat) (i : CIndex r)
    (hi : active r.parameters (instanceState pid s) i=Bool.true) :
    outputOffset r.parameters (instanceState pid s) i<r.rows*r.columns := by
  have hh : row r.parameters (instanceState pid s) i.1<r.rows ∧ col r.parameters (instanceState pid s) i.2.1<r.columns := by
    simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hi
  rw [source_offset]
  exact output_bound r _ _ hh.1 hh.2

theorem padded_a_bound (r : Request) (s : State Word) (pid k : Nat) (hp : pid<programs r)
    (hk : k≤count r.parameters+2) (i : TileIndex [r.blockRows,r.blockInner]) :
    aOffset r.parameters (instanceState pid s) k i≤envelopeA r := by
  have hr := Nat.mod_lt (row r.parameters (instanceState pid s) i.1) (nonempty_program r pid hp).1
  have hj : i.2.1.val<r.blockInner := i.2.1.isLt
  have hm := Nat.mul_le_mul_right r.blockInner hk
  have hh : k*r.blockInner+i.2.1.val≤(count r.parameters+3)*r.blockInner := by nlinarith
  rw [a_offset]
  exact Nat.add_le_add (Nat.mul_le_mul_right _ (Nat.le_of_lt hr)) hh

theorem padded_b_bound (r : Request) (s : State Word) (pid k : Nat) (hp : pid<programs r)
    (hk : k≤count r.parameters+2) (i : TileIndex [r.blockInner,r.blockColumns]) :
    bOffset r.parameters (instanceState pid s) k i≤envelopeB r := by
  have hc := Nat.mod_lt (col r.parameters (instanceState pid s) i.2.1) (nonempty_program r pid hp).2
  have hj : i.1.val<r.blockInner := i.1.isLt
  have hm := Nat.mul_le_mul_right r.blockInner hk
  have hh : k*r.blockInner+i.1.val≤(count r.parameters+3)*r.blockInner := by nlinarith
  rw [b_offset]
  exact Nat.add_le_add (Nat.mul_le_mul_right _ hh) (Nat.le_of_lt hc)

theorem padded_c_bound (r : Request) (s : State Word) (pid : Nat) (hp : pid<programs r) (i : CIndex r) :
    outputOffset r.parameters (instanceState pid s) i≤envelopeC r := by
  have hc : 0<tileColumns r := by unfold programs at hp; nlinarith
  have hr : pid/tileColumns r<tileRows r := (Nat.div_lt_iff_lt_mul hc).mpr hp
  have hn := Nat.mod_lt pid hc
  have hrow : row r.parameters (instanceState pid s) i.1≤tileRows r*r.blockRows := by
    have hmul := Nat.mul_le_mul_right r.blockRows (Nat.succ_le_of_lt hr)
    have hi : i.1.val<r.blockRows := i.1.isLt
    change (pid/tileColumns r)*r.blockRows+i.1.val≤_
    nlinarith
  have hcol : col r.parameters (instanceState pid s) i.2.1≤tileColumns r*r.blockColumns := by
    have hmul := Nat.mul_le_mul_right r.blockColumns (Nat.succ_le_of_lt hn)
    have hi : i.2.1.val<r.blockColumns := i.2.1.isLt
    change (pid%tileColumns r)*r.blockColumns+i.2.1.val≤_
    nlinarith
  rw [source_offset]
  unfold offset envelopeC
  rw [Nat.mul_comm (max r.columns 1)]
  exact Nat.add_le_add (Nat.mul_le_mul_right _ hrow) hcol

/-- Load/store allocation bounds and finite source pointer arithmetic. -/
structure SourceSafety (r : Request) (s : State Word) (a : Allocation r s) : Prop where
  subtraction : ∀ k, k<count r.parameters → k*r.blockInner<r.inner ∧
    (r.inner : Int)-(k*r.blockInner : Nat)=(r.inner-k*r.blockInner : Nat)
  loadA : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockRows,r.blockInner],
    i.2.1.val<r.inner-k*r.blockInner → aOffset r.parameters (instanceState pid s) k i<r.rows*r.inner
  loadB : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockInner,r.blockColumns],
    i.1.val<r.inner-k*r.blockInner → bOffset r.parameters (instanceState pid s) k i<r.inner*r.columns
  store : ∀ pid i, active r.parameters (instanceState pid s) i=Bool.true →
    outputOffset r.parameters (instanceState pid s) i<r.rows*r.columns
  pointerA : ∀ pid k, pid<programs r → k≤count r.parameters+2 → ∀ i : TileIndex [r.blockRows,r.blockInner],
    aOffset r.parameters (instanceState pid s) k i<2^31 ∧
      a.baseA+4*aOffset r.parameters (instanceState pid s) k i<2^64
  pointerB : ∀ pid k, pid<programs r → k≤count r.parameters+2 → ∀ i : TileIndex [r.blockInner,r.blockColumns],
    bOffset r.parameters (instanceState pid s) k i<2^31 ∧
      a.baseB+4*bOffset r.parameters (instanceState pid s) k i<2^64
  pointerC : ∀ pid, pid<programs r → ∀ i : CIndex r,
    outputOffset r.parameters (instanceState pid s) i<2^31 ∧
      a.baseC+4*outputOffset r.parameters (instanceState pid s) i<2^64

theorem source_safety (r : Request) (s : State Word) (a : Allocation r s) (d : IndexDomain r s a) :
    SourceSafety r s a where
  subtraction := source_subtraction r
  loadA := loadA_bound r s
  loadB := loadB_bound r s
  store := store_bound r s
  pointerA := by
    intro pid k hp hk i
    have h := padded_a_bound r s pid k hp hk i
    exact ⟨lt_of_le_of_lt h d.addressA,lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 h) _) d.byteA⟩
  pointerB := by
    intro pid k hp hk i
    have h := padded_b_bound r s pid k hp hk i
    exact ⟨lt_of_le_of_lt h d.addressB,lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 h) _) d.byteB⟩
  pointerC := by
    intro pid hp i
    have h := padded_c_bound r s pid hp i
    exact ⟨lt_of_le_of_lt h d.addressC,lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 h) _) d.byteC⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
