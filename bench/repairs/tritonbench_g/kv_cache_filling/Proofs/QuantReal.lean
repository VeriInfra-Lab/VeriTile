import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SharedTarget
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantReal
open QuantProfile
noncomputable section

def low (n : Nat) (x : Fin n → ℝ) : ℝ :=
  if h : 0 < n then Finset.univ.inf' ⟨⟨0,h⟩, Finset.mem_univ _⟩ x else 0

def high (n : Nat) (x : Fin n → ℝ) : ℝ :=
  if h : 0 < n then Finset.univ.sup' ⟨⟨0,h⟩, Finset.mem_univ _⟩ x else 0

def ops : Ops ℝ where
  literal := id
  add := (· + ·)
  sub := (· - ·)
  divide := (· / ·)
  minimum := min
  maximum := max
  equal := fun x y => decide (x = y)
  byte := fun x => BitVec.ofNat 8 (Int.floor x).toNat
  low := low
  high := high

theorem ops_low (n : Nat) (x : Fin n → ℝ) : ops.low n x = low n x := rfl
theorem ops_high (n : Nat) (x : Fin n → ℝ) : ops.high n x = high n x := rfl
theorem ops_minimum (x y : ℝ) : ops.minimum x y = min x y := rfl
theorem ops_maximum (x y : ℝ) : ops.maximum x y = max x y := rfl

theorem low_le {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (i : Fin n) : low n x ≤ x i := by
  simp only [low, dif_pos hn]
  exact Finset.inf'_le _ (Finset.mem_univ i)

theorem le_low {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (a : ℝ)
    (h : ∀ i, a ≤ x i) : a ≤ low n x := by
  simp only [low, dif_pos hn]
  exact Finset.le_inf' _ _ (fun i _ => h i)

theorem le_high {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (i : Fin n) : x i ≤ high n x := by
  simp only [high, dif_pos hn]
  exact Finset.le_sup' _ (Finset.mem_univ i)

theorem high_le {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (a : ℝ)
    (h : ∀ i, x i ≤ a) : high n x ≤ a := by
  simp only [high, dif_pos hn]
  exact Finset.sup'_le _ _ (fun i _ => h i)

theorem extrema (n : Nat) (hn : 0 < n) (x : Nat → ℝ) :
    Extrema n x (low n (fun i => x i.val)) (high n (fun i => x i.val)) := by
  refine ⟨fun i hi => low_le hn (fun j => x j.val) ⟨i,hi⟩,
    fun i hi => le_high hn (fun j => x j.val) ⟨i,hi⟩, ?_, ?_⟩
  · obtain ⟨i,_,he⟩ := Finset.exists_mem_eq_inf' (s := (Finset.univ : Finset (Fin n)))
      ⟨⟨0,hn⟩,Finset.mem_univ _⟩ (fun i => x i.val)
    exact ⟨i.val,i.isLt,by simpa only [low,dif_pos hn] using he.symm⟩
  · obtain ⟨i,_,he⟩ := Finset.exists_mem_eq_sup' (s := (Finset.univ : Finset (Fin n)))
      ⟨⟨0,hn⟩,Finset.mem_univ _⟩ (fun i => x i.val)
    exact ⟨i.val,i.isLt,by simpa only [high,dif_pos hn] using he.symm⟩

theorem pad_extrema {w n : Nat} (pos : 0 < w) (covered : w ≤ n)
    (x : Nat → ℝ) (lo hi : ℝ) (h : Extrema w x lo hi) :
    low n (pad w n x) = lo ∧ high n (pad w n x) = hi := by
  have hn : 0 < n := by omega
  constructor
  · apply le_antisymm
    · obtain ⟨i,hin,he⟩ := h.attains_low
      simpa [pad,hin,he] using low_le hn (pad w n x) ⟨i,by omega⟩
    · apply le_low hn
      intro i
      apply h.lower
      split <;> omega
  · apply le_antisymm
    · apply high_le hn
      intro i
      apply h.upper
      split <;> omega
    · obtain ⟨i,hin,he⟩ := h.attains_high
      simpa [pad,hin,he] using le_high hn (pad w n x) ⟨i,by omega⟩

/-- Both halves participate in INT4 extrema; replicated padding cannot change
an extremum of the complete logical input head. -/
theorem pair_pad_extrema {w n : Nat} (pos : 0 < w) (covered : w ≤ n)
    (x : Nat → ℝ) (lo hi : ℝ) (h : Extrema (2*w) x lo hi) :
    low n (fun i => min (pad w n x i) (pad w n (fun j => x (w+j)) i)) = lo ∧
    high n (fun i => max (pad w n x i) (pad w n (fun j => x (w+j)) i)) = hi := by
  have hn : 0 < n := by omega
  have ib (i : Fin n) : (if i.val < w then i.val else 0) < w := by split <;> omega
  constructor
  · apply le_antisymm
    · obtain ⟨i,hin,he⟩ := h.attains_low
      by_cases iw : i < w
      · have a := low_le hn (fun j => min (pad w n x j) (pad w n (fun k => x (w+k)) j))
          (⟨i,by omega⟩ : Fin n)
        exact le_trans a (by simpa [pad,iw,he] using min_le_left (x i) (x (w+i)))
      · have b : i-w < w := by omega
        have a := low_le hn (fun j => min (pad w n x j) (pad w n (fun k => x (w+k)) j))
          (⟨i-w,by omega⟩ : Fin n)
        exact le_trans a (by simpa [pad,b,Nat.add_sub_of_le (by omega : w ≤ i),he] using min_le_right (x (i-w)) (x i))
    · apply le_low hn
      intro i
      exact le_min (h.lower _ (by have := ib i; omega)) (h.lower _ (by have := ib i; omega))
  · apply le_antisymm
    · apply high_le hn
      intro i
      exact max_le (h.upper _ (by have := ib i; omega)) (h.upper _ (by have := ib i; omega))
    · obtain ⟨i,hin,he⟩ := h.attains_high
      by_cases iw : i < w
      · have a := le_high hn (fun j => max (pad w n x j) (pad w n (fun k => x (w+k)) j))
          (⟨i,by omega⟩ : Fin n)
        exact le_trans (by simpa [pad,iw,he] using le_max_left (x i) (x (w+i))) a
      · have b : i-w < w := by omega
        have a := le_high hn (fun j => max (pad w n x j) (pad w n (fun k => x (w+k)) j))
          (⟨i-w,by omega⟩ : Fin n)
        exact le_trans (by simpa [pad,b,Nat.add_sub_of_le (by omega : w ≤ i),he] using le_max_right (x (i-w)) (x i)) a

theorem scale_real (levels : Nat) (lo hi : ℝ) :
    scale ops levels lo hi = idealScale levels lo hi := by
  simp [scale,ops,idealScale]

theorem zero_real (levels : Nat) (lo hi : ℝ) :
    zero ops levels lo hi = -lo / idealScale levels lo hi := by
  simp only [zero,scale_real]
  simp [ops]

theorem code_real (levels : Nat) (lo hi x : ℝ) :
    code ops levels lo hi x = idealCode levels lo hi x := by
  simp only [code,scale_real,zero_real]
  simp [ops,idealCode,add_div,sub_eq_add_neg]

#axiomsClean pair_pad_extrema
#axiomsClean code_real

end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantReal

namespace VeriTile.Bench.Repairs.KvCacheFilling.SharedTarget
open QuantProfile QuantReal
noncomputable section

/-- Independent full-head mathematical result. Extrema use exactly dk/dv
logical inputs; block size and replicated padding do not occur here. -/
def idealValue (r : Request) (k v : Nat → ℝ) : Address → Cell ℝ (BitVec 8) :=
  rowValue
    (idealRow r.policy r.ck k (low r.dk (fun i => k i.val)) (high r.dk (fun i => k i.val)))
    (idealRow r.policy r.cv v (low r.dv (fun i => v i.val)) (high r.dv (fun i => v i.val)))

theorem real_row_independent (policy : Policy) (w n : Nat) (pos : 0 < w)
    (covered : w ≤ n) (x : Nat → ℝ) :
    row ops policy w n x = idealRow policy w x
      (low (if policy = .int4 then 2*w else w) (fun i => x i.val))
      (high (if policy = .int4 then 2*w else w) (fun i => x i.val)) := by
  cases policy with
  | copy => rfl
  | int8 =>
    have he := pad_extrema pos covered x _ _ (extrema w pos x)
    simp only [row,row8,ops_low,ops_high,he.1,he.2]
    change _ = idealRow .int8 w x (low w (fun i => x i.val)) (high w (fun i => x i.val))
    simp only [idealRow,scale_real,zero_real,code_real]
  | int4 =>
    have he := pair_pad_extrema pos covered x _ _ (extrema (2*w) (by omega) x)
    simp only [row,row4,ops_low,ops_high,ops_minimum,ops_maximum,he.1,he.2]
    change _ = idealRow .int4 w x (low (2*w) (fun i => x i.val)) (high (2*w) (fun i => x i.val))
    simp only [idealRow,scale_real,zero_real,code_real,packByte,Nat.add_comm]

theorem real_value_independent (r : Request) (d : Domain r) (k v : Nat → ℝ) (address : Address) :
    value ops r k v address = idealValue r k v address := by
  have pos := packed_widths_positive r d
  have ks : r.dk = if r.policy = .int4 then 2*r.ck else r.ck := by
    simpa [Request.factor,ite_mul] using d.key_shape
  have vs : r.dv = if r.policy = .int4 then 2*r.cv else r.cv := by
    simpa [Request.factor,ite_mul] using d.value_shape
  unfold value idealValue
  rw [real_row_independent r.policy r.ck _ pos.1 (block_covers _),
    real_row_independent r.policy r.cv _ pos.2 (block_covers _),ks,vs]

#axiomsClean real_value_independent
end
end VeriTile.Bench.Repairs.KvCacheFilling.SharedTarget
