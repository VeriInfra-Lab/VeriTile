import bench.tritonbench_g.streamk_matmul.Proofs.Intervals
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Arithmetic
open Source Plan
set_option maxHeartbeats 12000000
noncomputable section

def sum (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => sum f n+f n

theorem sum_congr (f g : Nat → ℝ) (n : Nat) (h : ∀ k, k < n → f k=g k) : sum f n=sum g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sum,ih (fun k hk=>h k (by omega)),h n (by omega)]

theorem sum_finset (f : Nat → ℝ) (n : Nat) : sum f n=∑ k ∈ Finset.range n,f k := by
  induction n with
  | zero => simp [sum]
  | succ n ih => simp [sum,Finset.sum_range_succ,ih]

theorem sum_fin (f : Nat → ℝ) (n : Nat) : (∑ k : Fin n,f k.val)=sum f n := by
  rw [sum_finset,Finset.sum_range]

theorem range_fold (f : Nat → ℝ) (n : Nat) (z : ℝ) :
    (List.range n).foldl (fun z k=>z+f k) z=z+sum f n := by
  induction n with
  | zero => simp [sum]
  | succ n ih => simp [List.range_succ,List.foldl_append,ih,sum,add_assoc]

theorem fin_fold (f : Fin n → ℝ) (z : ℝ) :
    (List.finRange n).foldl (fun z k=>z+f k) z=z+∑ k,f k := by
  rw [Fin.sum_univ_def]
  have aux:∀ xs : List (Fin n), xs.foldl (fun z k=>z+f k) z=z+(xs.map f).sum := by
    intro xs
    induction xs generalizing z with
    | nil => simp
    | cons x xs ih => simp [ih,add_assoc]
  exact aux _

theorem sum_add (f : Nat → ℝ) (a b : Nat) : sum f (a+b)=sum f a+sum (fun k=>f (a+k)) b := by
  induction b with
  | zero => simp [sum]
  | succ b ih => simp [sum,ih,Nat.add_assoc,add_assoc]

theorem interval_fold (f : Nat → ℝ) (lo hi : Nat) (h : lo ≤ hi) (z : ℝ) :
    (List.range' lo (hi-lo)).foldl (fun z k=>z+f k) z=z+(sum f hi-sum f lo) := by
  rw [List.range'_eq_map_range,List.foldl_map,range_fold]
  have hs:=sum_add f lo (hi-lo)
  rw [Nat.add_sub_of_le h] at hs
  linarith

theorem sum_padding (f : Nat → ℝ) (K n : Nat) :
    sum (fun k=>if k < K then f k else 0) n=sum f (min K n) := by
  induction n with
  | zero => simp [sum]
  | succ n ih =>
    by_cases hn:n < K
    · have hm:min K n=n:=min_eq_right (by omega)
      have hnext:min K (n+1)=n+1:=min_eq_right (by omega)
      simp [sum,ih,hn,hm,hnext]
    · have hm:min K n=K:=min_eq_left (by omega)
      have hnext:min K (n+1)=K:=min_eq_left (by omega)
      simp [sum,ih,hn,hm,hnext]

theorem grouped_sum (f : Nat → ℝ) (B n : Nat) :
    sum (fun block=>sum (fun k=>f (block*B+k)) B) n=sum f (n*B) := by
  induction n with
  | zero => simp [sum]
  | succ n ih => simp only [sum,ih,Nat.succ_mul,sum_add]

def real : Instructions ℝ := ⟨0,fun a b seed=>seed+∑ k : Fin 8,a k*b k,id,(·+·)⟩

def term (p : Parameters) (a : A p → ℝ) (b : B p → ℝ) (tile : Nat) (lane : Lane p) (k : Nat) : ℝ :=
  loadA p 0 a (row p tile lane) k*loadB p 0 b k (col p tile lane)

def iteration (p : Parameters) (a : A p → ℝ) (b : B p → ℝ) (tile : Nat) (lane : Lane p) (iter : Nat) : ℝ :=
  sum (fun k=>term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+k)) p.h.bk

theorem native_block_length (p : Parameters) : ((p.h.bk+7)/8)*8=p.h.bk := by
  have hd:=Nat.div_add_mod p.h.bk 8
  have ht:=Nat.div_add_mod (p.h.bk+7) 8
  have hm:=Nat.mod_lt (p.h.bk+7) (by decide : 0 < 8)
  have hn:=p.nativeK
  omega

theorem block_real (p : Parameters) (a : A p → ℝ) (b : B p → ℝ)
    (tile iter : Nat) (lane : Lane p) (seed : ℝ) :
    block p real a b tile iter lane seed=seed+iteration p a b tile lane iter := by
  unfold block
  change (List.range ((p.h.bk+7)/8)).foldl (fun acc group=>
    acc+∑ j : Fin 8,term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+group*8+j.val)) seed=_
  have hsum:∀ group, (∑ j : Fin 8,term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+group*8+j.val))=
      sum (fun j=>term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+group*8+j)) 8 := by
    intro group
    exact sum_fin (fun j=>term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+group*8+j)) 8
  simp_rw [hsum]
  rw [range_fold]
  unfold iteration
  have he:sum (fun group=>sum (fun j=>term p a b tile lane
      ((iter%(select p.h).iterations)*p.h.bk+group*8+j)) 8) ((p.h.bk+7)/8)=
      sum (fun j=>term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+j)) p.h.bk := by
    simpa only [Nat.add_assoc,native_block_length p] using
      grouped_sum (fun j=>term p a b tile lane ((iter%(select p.h).iterations)*p.h.bk+j)) 8 ((p.h.bk+7)/8)
  rw [he]

theorem chunk_real (p : Parameters) (a : A p → ℝ) (b : B p → ℝ)
    (tile lo hi : Nat) (hh : lo ≤ hi) (lane : Lane p) :
    chunkValue p real a b tile lo hi lane=
      sum (iteration p a b tile lane) hi-sum (iteration p a b tile lane) lo := by
  unfold chunkValue
  simp_rw [block_real]
  rw [interval_fold _ lo hi hh]
  simp [real]

theorem term_padding (p : Parameters) (a : A p → ℝ) (b : B p → ℝ)
    (tile : Nat) (lane : Lane p) (k : Nat) (hk : p.h.k ≤ k) : term p a b tile lane k=0 := by
  simp [term,loadA,loadB,show ¬k < p.h.k by omega]

/-- Exact Real meaning of the complete per-tile K loop, with all masked tail
operands removed and the target sum taken over the original K dimension. -/
theorem full_real (p : Parameters) (a : A p → ℝ) (b : B p → ℝ)
    (tile : Nat) (lane : Lane p) (ha : active p tile lane) :
    chunkValue p real a b tile 0 (select p.h).iterations lane=
      Reference.matrixProduct p a b ⟨row p tile lane,ha.1⟩ ⟨col p tile lane,ha.2⟩ := by
  rw [chunk_real p a b tile 0 _ (Nat.zero_le _) lane]
  simp only [sum,sub_zero]
  have hiter:sum (iteration p a b tile lane) (select p.h).iterations=
      sum (fun iter=>sum (fun k=>term p a b tile lane (iter*p.h.bk+k)) p.h.bk) (select p.h).iterations := by
    apply sum_congr
    intro iter hi
    simp only [iteration,Nat.mod_eq_of_lt hi]
  rw [hiter,grouped_sum]
  have hcover:p.h.k ≤ (select p.h).iterations*p.h.bk:=by
    rw [selected_iterations]
    exact IntScaledMatmul.Revision001.LogicalSum.ceil_cover p.h.k p.h.bk p.blockK
  have hpad:sum (term p a b tile lane) ((select p.h).iterations*p.h.bk)=
      sum (fun k=>if k < p.h.k then term p a b tile lane k else 0) ((select p.h).iterations*p.h.bk) := by
    apply sum_congr
    intro k hk
    by_cases h:k < p.h.k
    · simp [h]
    · simp [h,term_padding p a b tile lane k (by omega)]
  rw [hpad,sum_padding,min_eq_left hcover,←sum_fin]
  apply Finset.sum_congr rfl
  intro k hk
  simp [term,loadA,loadB,ha.1,ha.2,k.isLt,Reference.matrixProduct]

#axiomsClean chunk_real
#axiomsClean full_real
end
end Arithmetic
end VeriTile.Bench.Repairs.StreamK.Whole001
