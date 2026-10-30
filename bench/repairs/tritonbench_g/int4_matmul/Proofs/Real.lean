import bench.repairs.tritonbench_g.int4_matmul.Proofs.Grid
import bench.repairs.tritonbench_g.int4_matmul.Proofs.RealSum
namespace VeriTile.Bench.Repairs.Int4Matmul
open RealSum
set_option maxHeartbeats 6000000
noncomputable section

def realProduct (s : Shape) (m : State ℝ) (r c k : Nat) :=
  logicalA s m r k*((logicalQ s m k c-logicalZ s m k c : Int):ℝ)*logicalScale s m k c

def paddedProduct (s : Shape) (m : State ℝ) (r c k : Nat) :=
  if k<s.k then realProduct s m r c k else 0

theorem dot8_real (s : Shape) (m : State ℝ) (r c base : Nat) (seed : ℝ) :
    realOps.dot8
      (fun i => if base+i.val<s.k then logicalA s m r (base+i.val) else 0)
      (fun i => if base+i.val<s.k then dequantized s realOps m (base+i.val) c else 0)
      seed = seed+sumReal (fun i => paddedProduct s m r c (base+i)) 8 := by
  change seed+sumReal (fun i =>
    (if base+i%8<s.k then logicalA s m r (base+i%8) else 0)*
    (if base+i%8<s.k then dequantized s realOps m (base+i%8) c else 0)) 8 = _
  congr 1
  apply sum_congr
  intro k hk
  simp only [Nat.mod_eq_of_lt hk]
  unfold paddedProduct realProduct dequantized
  by_cases h : base+k<s.k
  · simp [h,realOps];ring
  · simp [h]

theorem referencePartial_real (s : Shape) (t : Tile) (ht : t.Valid) (m : State ℝ) (w : Work) :
    referencePartial s t realOps m w=
      sumReal (fun block => sumReal (fun lane => paddedProduct s m w.row w.col (kIndex t w.split block lane)) t.bk) (rounds s t) := by
  have bk : t.bk/8*8=t.bk := Nat.div_mul_cancel ht.bk8
  have block (b : Nat) (z : ℝ) :
      (List.range (t.bk/8)).foldl (fun seed chunk =>
        realOps.dot8
          (fun i => let k := (b*t.splits+w.split)*t.bk+chunk*8+i
            if k<s.k then logicalA s m w.row k else realOps.zero)
          (fun i => let k := (b*t.splits+w.split)*t.bk+chunk*8+i
            if k<s.k then dequantized s realOps m k w.col else realOps.mul (realOps.fromInt 0) realOps.zero)
          seed) z=
      z+sumReal (fun lane => paddedProduct s m w.row w.col (kIndex t w.split b lane)) t.bk := by
    simp only [show realOps.zero=0 from rfl,show realOps.mul (realOps.fromInt 0) 0=0 from by simp [realOps]]
    simp only [dot8_real,range_fold]
    have chunks := sum_chunks (fun lane => paddedProduct s m w.row w.col (kIndex t w.split b lane)) 8 (t.bk/8)
    rw [bk] at chunks
    rw [←chunks]
    congr 1
    apply sum_congr;intro j hj
    apply sum_congr;intro i hi
    congr 1
    rw [kIndex_eq]
    omega
  unfold referencePartial
  simp only [block]
  change (List.range (rounds s t)).foldl (fun acc b => acc+sumReal (fun lane => paddedProduct s m w.row w.col (kIndex t w.split b lane)) t.bk) 0 = _
  rw [range_fold,zero_add]

theorem all_split_real (s : Shape) (t : Tile) (ht : t.Valid) (m : State ℝ) (r c : Nat) :
    sumReal (fun sp => referencePartial s t realOps m ⟨r,c,sp⟩) t.splits=mathematical s m r c := by
  simp only [referencePartial_real s t ht]
  rw [sum_swap]
  have inner (b : Nat) :
      sumReal (fun sp => sumReal (fun e => paddedProduct s m r c (kIndex t sp b e)) t.bk) t.splits=
      sumReal (fun j => paddedProduct s m r c (b*(t.splits*t.bk)+j)) (t.splits*t.bk) := by
    rw [←sum_chunks (fun j => paddedProduct s m r c (b*(t.splits*t.bk)+j)) t.bk t.splits]
    apply sum_congr;intro sp hsp
    apply sum_congr;intro e he
    congr 1;unfold kIndex;ring
  simp only [inner,sum_chunks]
  have cover := ceil_cover s.k (t.bk*t.splits) (Nat.mul_pos ht.bkpos (split_positive t ht))
  have h : s.k≤rounds s t*(t.splits*t.bk) := by simpa [rounds,Nat.mul_comm] using cover
  unfold paddedProduct
  rw [sum_padding,Nat.min_eq_left h]
  rfl

def cellWorks (r c : Nat) (ws : List Work) := ws.filter (fun w => decide (w.row=r ∧ w.col=c))

theorem cellWorks_perm (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (ws : List Work) (complete : Complete s t ws) (r c : Nat) (hr : r<s.m) (hc : c<s.n) :
    (cellWorks r c ws).Perm ((List.range t.splits).map (fun sp => Work.mk r c sp)) := by
  have nodup : ws.Nodup := complete.nodup_iff.mpr (sourceSchedule_nodup s t hs ht)
  apply (List.perm_ext_iff_of_nodup (nodup.filter _) (List.nodup_range.map (by
    intro a b h
    exact congrArg Work.split h))).mpr
  intro w
  simp only [List.mem_filter,decide_eq_true_eq,List.mem_map,List.mem_range]
  rw [complete_active s t hs ht ws complete]
  constructor
  · rintro ⟨⟨_,_,hsp⟩,hrow,hcol⟩
    exact ⟨w.split,hsp,by cases w;simp_all⟩
  · rintro ⟨sp,hsp,rfl⟩
    exact ⟨⟨hr,hc,hsp⟩,rfl,rfl⟩

theorem referenceOutput_filter (s : Shape) (t : Tile) (p : Ops α) (initial : State α)
    (r c : Nat) (ws : List Work) (seed : α) :
    referenceOutput s t p initial r c ws seed=
      referenceOutput s t p initial r c (cellWorks r c ws) seed := by
  induction ws generalizing seed with
  | nil => rfl
  | cons w ws ih =>
    by_cases h : w.row=r ∧ w.col=c
    · simpa [referenceOutput,cellWorks,h] using ih _
    · simpa [referenceOutput,cellWorks,h] using ih seed

theorem referenceOutput_real_fold (s : Shape) (t : Tile) (hsplit : t.splits≠1)
    (m : State ℝ) (r c : Nat) (ws : List Work) (seed : ℝ) :
    referenceOutput s t realOps m r c ws seed=
      (cellWorks r c ws).foldl (fun acc w => acc+referencePartial s t realOps m w) seed := by
  induction ws generalizing seed with
  | nil => rfl
  | cons w ws ih =>
    by_cases h : w.row=r ∧ w.col=c
    · simpa [referenceOutput,cellWorks,h,hsplit,show realOps.atomicAdd=(·+·) from rfl] using
        ih (seed+referencePartial s t realOps m w)
    · simpa [referenceOutput,cellWorks,h] using ih seed

theorem whole_real (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (m : State ℝ) (ws : List Work) (complete : Complete s t ws)
    (r c : Nat) (hr : r<s.m) (hc : c<s.n) :
    (call s t realOps ws m).values .output (cAddress s r c)=mathematical s m r c := by
  rw [call_result s t hs ht realOps ws complete m r c hr hc]
  have perm := cellWorks_perm s t hs ht ws complete r c hr hc
  by_cases hsplit : t.splits=1
  · have one : cellWorks r c ws=[⟨r,c,0⟩] := by
      simpa [hsplit] using perm
    rw [referenceOutput_filter,one]
    simpa [referenceOutput,hsplit,sumReal] using all_split_real s t ht m r c
  · rw [referenceOutput_real_fold s t hsplit]
    rw [perm.foldl_eq' (fun _ _ _ _ _ => by ring)]
    rw [List.foldl_map,range_fold]
    change 0+sumReal (fun sp => referencePartial s t realOps m ⟨r,c,sp⟩) t.splits = _
    rw [zero_add,all_split_real s t ht]

#axiomsClean cellWorks_perm
#axiomsClean referenceOutput_filter
#axiomsClean referenceOutput_real_fold
#axiomsClean whole_real
#axiomsClean dot8_real
#axiomsClean referencePartial_real
#axiomsClean all_split_real
end
end VeriTile.Bench.Repairs.Int4Matmul
