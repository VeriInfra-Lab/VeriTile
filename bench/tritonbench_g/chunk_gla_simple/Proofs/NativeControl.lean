import bench.tritonbench_g.chunk_gla_simple.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeControl
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open Native (word)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

/-- Ordinary successful allocation/compiler/launch metadata. Failed allocation,
block setup or grid setup is a separate wrapper outcome. These are not new
input validation restrictions. Tensor blocks have at most 2^20 elements and
CUDA's accepted grid axes give nonnegative signed-i32 program IDs. -/
structure Accepted (c : Shape) : Prop where
  storage : Native.Fits c
  nonempty : 0<c.heads*c.T*c.V
  block : c.BT≤2^20
  grid : ∀ q : FPGrid.Pid c, (launchIndex c q).value<2^31 ∧
    (launchIndex c q).chunk<2^31 ∧ (launchIndex c q).head<2^31

theorem block32 (c : Shape) (h : Accepted c) : c.BT<2^32 := lt_of_le_of_lt h.block (by decide)

def extend32 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64

theorem extend32_exact (n : Nat) (h : n<2^31) : extend32 n=word n := by
  apply BitVec.eq_of_toInt_eq
  rw [extend32,BitVec.toInt_signExtend_of_le (by decide),Native.word_exact n (lt_trans h (by decide))]
  have hn : (BitVec.ofNat 32 n).toNat=n := Nat.mod_eq_of_lt (lt_trans h (by decide))
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; norm_num at h ⊢; omega),hn]

/-- Actual program_id/arange sign extensions agree with the words used by
NativeIO, including the signed-i32 causal comparison. -/
theorem setup (c : Shape) (h : Accepted c) (q : FPGrid.Pid c) :
    extend32 (launchIndex c q).value=word (launchIndex c q).value ∧
    extend32 (launchIndex c q).chunk=word (launchIndex c q).chunk ∧
    extend32 (launchIndex c q).head=word (launchIndex c q).head ∧
    (∀ i : Fin c.BT, extend32 i.val=word i.val) ∧
    (∀ i : Fin c.BK, extend32 i.val=word i.val) ∧
    (∀ i : Fin c.BV, extend32 i.val=word i.val) := by
  have hb : c.BT<2^31 := lt_of_le_of_lt h.block (by decide)
  exact ⟨extend32_exact _ (h.grid q).1,extend32_exact _ (h.grid q).2.1,
    extend32_exact _ (h.grid q).2.2,fun i => extend32_exact _ (lt_trans i.isLt hb),
    fun i => extend32_exact _ (lt_of_lt_of_le i.isLt (le_trans c.bk_max (by decide))),
    fun i => extend32_exact _ (lt_of_lt_of_le i.isLt (le_trans (Nat.min_le_right _ _) (by decide)))⟩

theorem causal (c : Shape) (h : Accepted c) (i j : Fin c.BT) :
    (BitVec.ofNat 32 j.val).toInt ≤ (BitVec.ofNat 32 i.val).toInt ↔ j.val ≤ i.val := by
  have he (k : Fin c.BT) : (BitVec.ofNat 32 k.val).toInt=(k.val : Int) := by
    have hn := congrArg BitVec.toInt (extend32_exact k.val
      (lt_of_lt_of_le k.isLt (le_trans h.block (by decide))))
    rw [extend32,BitVec.toInt_signExtend_of_le (by decide),Native.word_exact _
      (lt_of_lt_of_le k.isLt (le_trans h.block (by decide)))] at hn
    exact hn
  rw [he i,he j]
  exact Int.ofNat_le

/-- Fuel is the finite bound of the source for-loop, not an unrolling limit.
Both the comparison and the backedge use the actual signed-i64 words. -/
def range {β : Type} (step : BitVec 64 → β → β) : Nat → BitVec 64 → BitVec 64 → β → β
  | 0,_,_,a => a
  | fuel+1,cur,stop,a =>
      if cur.toInt<stop.toInt then range step fuel (cur+word 1) stop (step cur a) else a

theorem range_eq {β : Type} (step : BitVec 64 → β → β) (natStep : Nat → β → β)
    (cur stop : Nat) (hs : stop<2^63) (hc : cur≤stop)
    (he : ∀ n, cur≤n → n<stop → step (word n)=natStep n) (a : β) :
    range step (stop-cur) (word cur) (word stop) a=foldRange natStep cur stop 1 a := by
  by_cases hl : cur<stop
  · have hn : stop-cur=(stop-(cur+1))+1 := by omega
    have hcmp : (word cur).toInt<(word stop).toInt :=
      (Native.comparison_exact cur stop (lt_trans hl hs) hs).2 hl
    rw [hn,range,if_pos hcmp,he cur (by omega) hl]
    have hadd : word cur+word 1=word (cur+1) := (BitVec.ofNat_add cur 1).symm
    rw [hadd,range_eq step natStep (cur+1) stop hs (by omega)
      (fun n h₁ h₂ => he n (by omega) h₂)]
    conv_rhs => rw [foldRange]
    simp [hl]
  · have hz : cur=stop := by omega
    subst cur
    rw [Nat.sub_self,range,foldRange]
    simp
termination_by stop-cur

/-- Every tested K-loop comparison and the final increment are exact. The
cast added to the source keeps the upper bound signed-i64 across [2^31,2^32). -/
theorem loop_control (c : Shape) (h : Accepted c) :
    let stop := (c.K+c.BK-1)/c.BK
    (word stop).toInt=(stop : Int) ∧
    (∀ n, n<stop → (word n).toInt<(word stop).toInt ∧
      (word n+word 1).toInt=(n+1 : Nat)) ∧
    ¬(word stop).toInt<(word stop).toInt := by
  dsimp only
  have hs := (Native.loop_bound c h.storage h.nonempty).2
  refine ⟨Native.word_exact _ hs,?_,lt_irrefl _⟩
  intro n hn
  refine ⟨(Native.comparison_exact _ _ (lt_trans hn hs) hs).2 hn,?_⟩
  rw [←BitVec.ofNat_add]
  exact Native.word_exact _ (lt_of_le_of_lt (Nat.succ_le_of_lt hn) hs)

#axiomsClean setup
#axiomsClean range_eq
#axiomsClean loop_control
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeControl
