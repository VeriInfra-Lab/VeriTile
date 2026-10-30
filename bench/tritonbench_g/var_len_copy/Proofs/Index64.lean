import bench.tritonbench_g.var_len_copy.Contract
import bench.tritonbench_g.var_len_copy.Proofs.Mathematical_Math

namespace VeriTile.Bench.Repairs.VarLenCopy
open VeriTile.Bench.TritonBenchG.VarLenCopy
noncomputable section
set_option maxHeartbeats 2000000

theorem signed64_exact (n : Int) (lo : -(2^63) ≤ n) (hi : n < 2^63) :
    signed64 n=n := BitVec.toInt_ofInt_eq_self (by decide) lo hi

theorem extend32 (w : Word) : signed64 w.toInt=w.toInt := by
  apply signed64_exact
  · have := BitVec.le_toInt w; norm_num at this ⊢; omega
  · have := BitVec.toInt_lt (x:=w); norm_num at this ⊢; omega

theorem pointer_exact (base o size : Nat) (capacity : base+4*size≤2^64) (bound : o<size) :
    pointer base o=addr base o := by
  have h : (base : Int)+4*(o : Int)<2^64 := by omega
  have hn : 0≤(base : Int)+4*(o : Int) := by omega
  simp only [pointer,BitVec.toNat_ofInt,Nat.cast_pow,Nat.cast_ofNat]
  rw [Int.emod_eq_of_lt hn h]
  change ((base+4*o : Nat) : Int).toNat=addr base o
  simp only [Int.toNat_natCast,addr]

theorem counter_bound (a : Segment) (c lane : Nat)
    (hc : c<rounds a.length.toInt 256) (hl : lane<256) :
    c*256<a.len ∧ c*256+lane<2^31+256 ∧ (c+1)*256<2^31+256 := by
  have hi := (Math.loop_test a.length.toInt (by decide : 0<256)).mpr hc
  have hm := BitVec.toInt_lt (x:=a.length)
  change c*256<a.length.toInt.toNat at hi
  unfold Segment.len
  norm_num at hm ⊢
  omega

theorem lane_exact (l : Layout) (r : Request) (s : Memory) (prep : Prepared l r s)
    (valid : Valid r) (p c lane : Nat) (hp : p<r.count)
    (hc : c<rounds (r.segment p).length.toInt 256) (hl : lane<256) :
    laneEvent l (r.segment p) c lane =
      if c*256+lane<(r.segment p).len then
        some (addr l.output ((r.segment p).dest+c*256+lane),
              addr l.source ((r.segment p).old+c*256+lane)) else none := by
  let a := r.segment p
  have hb := counter_bound a c lane hc hl
  have ci : signed64 ((c*256 : Nat) : Int)=((c*256 : Nat) : Int) := signed64_exact _ (by omega) (by omega)
  have li : signed64 lane=(lane : Int) := signed64_exact _ (by omega) (by omega)
  have oi : signed64 (((c*256 : Nat) : Int)+(lane : Int))=((c*256+lane : Nat) : Int) :=
    by simpa only [Nat.cast_add] using signed64_exact ((c*256+lane : Nat) : Int) (by omega) (by omega)
  simp only [laneEvent,ci,li,oi,extend32]
  by_cases ha : c*256+lane<a.len
  · have hlen : 0<a.len := by omega
    obtain ⟨hos,hosb⟩ := valid.sourceBound p hp hlen
    obtain ⟨hns,hnsb⟩ := valid.outputBound p hp hlen
    change 0≤a.oldStart.toInt at hos
    change 0≤a.newStart.toInt at hns
    change a.old+a.len≤r.sourceSize at hosb
    change a.dest+a.len≤r.outputSize at hnsb
    have hmask : ((c*256+lane : Nat) : Int)<a.length.toInt := by
      change c*256+lane<a.length.toInt.toNat at ha
      omega
    have os : a.oldStart.toInt=(a.old : Int) := by simp only [Segment.old]; omega
    have ns : a.newStart.toInt=(a.dest : Int) := by simp only [Segment.dest]; omega
    have bo := BitVec.toInt_lt (x:=a.oldStart)
    have bn := BitVec.toInt_lt (x:=a.newStart)
    have so : signed64 (a.oldStart.toInt+((c*256+lane : Nat) : Int))=
        ((a.old+c*256+lane : Nat) : Int) := by
      have hsmall : signed64 (a.oldStart.toInt+((c*256+lane : Nat) : Int))=
          a.oldStart.toInt+((c*256+lane : Nat) : Int) :=
        signed64_exact _ (by norm_num at bo bn ⊢;omega) (by norm_num at bo bn ⊢;omega)
      rw [hsmall,os]
      push_cast
      omega
    have sn : signed64 (a.newStart.toInt+((c*256+lane : Nat) : Int))=
        ((a.dest+c*256+lane : Nat) : Int) := by
      have hsmall : signed64 (a.newStart.toInt+((c*256+lane : Nat) : Int))=
          a.newStart.toInt+((c*256+lane : Nat) : Int) :=
        signed64_exact _ (by norm_num at bo bn ⊢;omega) (by norm_num at bo bn ⊢;omega)
      rw [hsmall,ns]
      push_cast
      omega
    rw [if_pos hmask,so,sn,if_pos ha]
    rw [pointer_exact l.output _ l.outputSize prep.outputCapacity (by rw [prep.outputSize];omega),
      pointer_exact l.source _ l.sourceSize prep.sourceCapacity (by rw [prep.sourceSize];omega)]
  · have hmask : ¬((c*256+lane : Nat) : Int)<a.length.toInt := by
      change ¬c*256+lane<a.length.toInt.toNat at ha
      omega
    rw [if_neg hmask,if_neg ha]

theorem metadata_load (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (p : Nat) (hp : p<r.count) : loadSegment l s p=r.segment p := by
  have hp' : p<l.count := by rwa [prep.counts]
  have hpi : signed64 p=(p : Int) := signed64_exact _ (by omega) (by have := prep.grid;omega)
  simp only [loadSegment,hpi]
  rw [pointer_exact _ _ _ prep.oldCapacity hp',pointer_exact _ _ _ prep.lengthCapacity hp',
    pointer_exact _ _ _ prep.newCapacity hp',prep.oldCells p hp',prep.lengthCells p hp',prep.newCells p hp']

theorem events_mem (l : Layout) (r : Request) (s : Memory) (prep : Prepared l r s)
    (valid : Valid r) (p : Nat) (hp : p<r.count) (e : Event) :
    e∈events l (r.segment p) ↔ ∃ i, i<(r.segment p).len ∧
      e=(addr l.output ((r.segment p).dest+i),addr l.source ((r.segment p).old+i)) := by
  simp only [events,List.mem_flatMap,List.mem_range,List.mem_filterMap]
  constructor
  · rintro ⟨c,hc,lane,hl,he⟩
    rw [lane_exact l r s prep valid p c lane hp hc hl] at he
    split at he
    · rename_i h
      have he' := Option.some.inj he
      exact ⟨c*256+lane,h,by simpa [Nat.add_assoc] using he'.symm⟩
    · simp at he
  · rintro ⟨i,hi,rfl⟩
    have hl : i%256<256 := Nat.mod_lt _ (by decide)
    have he : i/256*256+i%256=i := by omega
    have hc : i/256<rounds (r.segment p).length.toInt 256 := by
      apply (Math.loop_test _ (by decide : 0<256)).mp
      change i/256*256<(r.segment p).len
      omega
    refine ⟨i/256,hc,i%256,hl,?_⟩
    rw [lane_exact l r s prep valid p _ _ hp hc hl,if_pos (by omega)]
    simp only [Nat.add_assoc,he]

/-- Includes the final increment; i32 payload boundaries do not constrain
allocation size or valid source/destination sums. -/
theorem former_i32_boundary : signed64 (2147483647+256)=2147483903 ∧
    signed64 (2147483392+256)=2147483648 := by decide

#axiomsClean lane_exact
#axiomsClean events_mem
#axiomsClean former_i32_boundary
end
end VeriTile.Bench.Repairs.VarLenCopy
