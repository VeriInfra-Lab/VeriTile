import bench.repairs.tritonbench_g.var_len_copy.Proofs.Index64

namespace VeriTile.Bench.Repairs.VarLenCopy.Native
open VeriTile.Bench.TritonBenchG.VarLenCopy
noncomputable section
set_option maxHeartbeats 3000000

/-- Includes the failed terminal test, not just the successful loop bodies. -/
theorem counter_fits (a : Segment) (c : Nat) (hc : c≤rounds a.length.toInt 256) :
    c*256<2^31+256 := by
  have hi := BitVec.toInt_lt (x:=a.length)
  have hm := Nat.div_mul_le_self (a.length.toInt.toNat+256-1) 256
  have hh := Nat.mul_le_mul_right 256 hc
  unfold rounds at hh
  norm_num at hi ⊢
  omega

theorem loop_test (a : Segment) (c : Nat) (hc : c≤rounds a.length.toInt 256) :
    signed64 ((c*256 : Nat) : Int)<signed64 a.length.toInt ↔ c<rounds a.length.toInt 256 := by
  have hb := counter_fits a c hc
  rw [signed64_exact _ (by omega) (by omega),extend32]
  have he := Math.loop_test a.length.toInt (by decide : 0<256) (c:=c)
  change c*256<a.length.toInt.toNat ↔ c < rounds a.length.toInt 256 at he
  omega

theorem increment (a : Segment) (c : Nat) (hc : c<rounds a.length.toInt 256) :
    signed64 (signed64 ((c*256 : Nat) : Int)+256)=(((c+1)*256 : Nat) : Int) := by
  have h0 := counter_fits a c (by omega)
  have h1 := counter_fits a (c+1) (by omega)
  have initial : signed64 ((c*256 : Nat) : Int)=((c*256 : Nat) : Int) :=
    signed64_exact _ (by omega) (by omega)
  rw [initial]
  have h : signed64 (((c*256 : Nat) : Int)+256)=((c*256 : Nat) : Int)+256 :=
    signed64_exact _ (by omega) (by omega)
  rw [h]
  push_cast
  omega

/-- All three active source-visible pointer additions remain inside the same
ordinary allocated object, even if their element sum exceeds signed i32. -/
theorem pointer_stages (base size start current lane : Nat)
    (capacity : base+4*size≤2^64) (active : start+current+lane<size) :
    pointer (pointer (pointer base start) current) lane=addr base (start+current+lane) := by
  have hs : start<size := by omega
  rw [pointer_exact base start size capacity hs]
  have cap2 : addr base start+4*(size-start)≤2^64 := by unfold addr;omega
  rw [pointer_exact _ current (size-start) cap2 (by omega)]
  have cap3 : addr (addr base start) current+4*(size-start-current)≤2^64 := by unfold addr;omega
  rw [pointer_exact _ lane (size-start-current) cap3 (by omega)]
  unfold addr
  omega

/-- Exact Python left-associated pointer chain, before compiler reassociation. -/
def stagedLane (l : Layout) (a : Segment) (c lane : Nat) : Option Event :=
  let i := signed64 ((c*256 : Nat) : Int)
  let lane64 := signed64 lane
  if signed64 (i+lane64)<signed64 a.length.toInt then
    some (pointer (pointer (pointer l.output (signed64 a.newStart.toInt)) i) lane64,
          pointer (pointer (pointer l.source (signed64 a.oldStart.toInt)) i) lane64)
  else none

theorem staged_lane (l : Layout) (r : Request) (s : Memory) (prep : Prepared l r s)
    (valid : Valid r) (p c lane : Nat) (hp : p<r.count)
    (hc : c<rounds (r.segment p).length.toInt 256) (hl : lane<256) :
    stagedLane l (r.segment p) c lane=laneEvent l (r.segment p) c lane := by
  rw [lane_exact l r s prep valid p c lane hp hc hl]
  let a := r.segment p
  have hb := counter_bound a c lane hc hl
  have ci : signed64 ((c*256 : Nat) : Int)=((c*256 : Nat) : Int) := signed64_exact _ (by omega) (by omega)
  have li : signed64 lane=(lane : Int) := signed64_exact _ (by omega) (by omega)
  have oi : signed64 (((c*256 : Nat) : Int)+(lane : Int))=((c*256+lane : Nat) : Int) := by
    simpa only [Nat.cast_add] using signed64_exact ((c*256+lane : Nat) : Int) (by omega) (by omega)
  simp only [stagedLane,ci,li,oi,extend32]
  by_cases active : c*256+lane<a.len
  · obtain ⟨hos,hs⟩ := valid.sourceBound p hp (by omega : 0<a.len)
    obtain ⟨hns,hn⟩ := valid.outputBound p hp (by omega : 0<a.len)
    change 0≤a.oldStart.toInt at hos
    change 0≤a.newStart.toInt at hns
    change a.old+a.len≤r.sourceSize at hs
    change a.dest+a.len≤r.outputSize at hn
    have os : a.oldStart.toInt=(a.old : Int) := by simp only [Segment.old];omega
    have ns : a.newStart.toInt=(a.dest : Int) := by simp only [Segment.dest];omega
    have hm : ((c*256+lane : Nat) : Int)<a.length.toInt := by
      change c*256+lane<a.length.toInt.toNat at active
      omega
    rw [if_pos hm,if_pos active,os,ns]
    rw [pointer_stages l.output l.outputSize _ _ _ prep.outputCapacity (by rw [prep.outputSize];omega),
      pointer_stages l.source l.sourceSize _ _ _ prep.sourceCapacity (by rw [prep.sourceSize];omega)]
  · have hm : ¬((c*256+lane : Nat) : Int)<a.length.toInt := by
      change ¬c*256+lane<a.length.toInt.toNat at active
      omega
    rw [if_neg hm,if_neg active]

def sourceEvents (l : Layout) (a : Segment) : List Event :=
  (List.range (rounds a.length.toInt 256)).flatMap fun c =>
    (List.range 256).filterMap (stagedLane l a c)

theorem source_events (l : Layout) (r : Request) (s : Memory) (prep : Prepared l r s)
    (valid : Valid r) (p : Nat) (hp : p<r.count) :
    sourceEvents l (r.segment p)=events l (r.segment p) := by
  apply List.flatMap_congr
  intro c hc
  apply List.filterMap_congr
  intro lane hl
  exact staged_lane l r s prep valid p c lane hp (List.mem_range.mp hc) (List.mem_range.mp hl)

#axiomsClean loop_test
#axiomsClean increment
#axiomsClean staged_lane
#axiomsClean source_events
end
end VeriTile.Bench.Repairs.VarLenCopy.Native
