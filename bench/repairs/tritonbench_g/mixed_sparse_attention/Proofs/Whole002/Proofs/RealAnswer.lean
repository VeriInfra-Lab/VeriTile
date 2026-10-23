import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Coverage
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def total (c : Config) (s : State R) (i : Fin c.BM) (f : Nat → ℝ) : ℝ :=
  (∑ b∈Finset.range (blockCount c s),blockSum c s i f b)+
    ∑ j∈Finset.range (columnCount c s),f (columnIndex c s j)

theorem final_consistent (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (i : Fin c.BM) (d : Fin c.BD) (hi : rowStart c s+i.val<sequenceLength c s) (hbn : 0<c.BN) :
    ∃ result,RowAgrees c i d (finalValues algebra dotModel c s) result ∧
      VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Consistent result
        (total c s i (fun j=>pow2 (logicalScore c s μ i j)))
        (total c s i (fun j=>pow2 (logicalScore c s μ i j)*vAt c s μ j d.val)) := by
  let mass : Nat → ℝ:=fun j=>pow2 (logicalScore c s μ i j)
  let value : Nat → ℝ:=fun j=>pow2 (logicalScore c s μ i j)*vAt c s μ j d.val
  have hm : ∀ start,start<blockCount c s → VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.denominator (tile c s μ Bool.false start i d)=
      chunk 1 (blockCount c s) start (blockSum c s i mass) := by
    intro start hs
    rw [tile_denominator,block_sum c s i start _ hs]
    simp [chunk,hs,mass]
  have ht : ∀ start,start<blockCount c s → VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.numerator (tile c s μ Bool.false start i d)=
      chunk 1 (blockCount c s) start (blockSum c s i value) := by
    intro start hs
    rw [tile_numerator,block_sum c s i start _ hs]
    simp [chunk,hs,value]
  obtain ⟨blocks,hblocks,hbc⟩:=loop_consistent c s μ h i d hi Bool.false (blockCount c s) 1 (by omega)
    (blockSum c s i mass) (blockSum c s i value) hm ht 0 0 0 (initialAccumulator algebra c)
    ⟨⊥,0,0⟩ (initial_agrees c i d) (by simpa [prefixSum] using VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.initial_consistent)
  simp only [zero_add] at hbc
  have hm' : ∀ start,start<columnCount c s → VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.denominator (tile c s μ Bool.true start i d)=
      chunk c.BN (columnCount c s) start (fun j=>mass (columnIndex c s j)) := by
    intro start hs
    rw [tile_denominator,column_sum c s i start _ hs]
  have ht' : ∀ start,start<columnCount c s → VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.numerator (tile c s μ Bool.true start i d)=
      chunk c.BN (columnCount c s) start (fun j=>value (columnIndex c s j)) := by
    intro start hs
    rw [tile_numerator,column_sum c s i start _ hs]
  obtain ⟨result,hr,hc⟩:=loop_consistent c s μ h i d hi Bool.true (columnCount c s) c.BN hbn
    (fun j=>mass (columnIndex c s j)) (fun j=>value (columnIndex c s j)) hm' ht'
    (∑ b∈Finset.range (blockCount c s),blockSum c s i mass b)
    (∑ b∈Finset.range (blockCount c s),blockSum c s i value b)
    0 _ blocks hblocks (by simpa [prefixSum] using hbc)
  exact ⟨result,hr,hc⟩

theorem list_range_sum (n : Nat) (f : Nat → ℝ) :
    ((List.range n).map f).sum=∑ j∈Finset.range n,f j := by
  induction n with
  | zero=>simp
  | succ n ih=>simp [List.range_succ,Finset.sum_range_succ,ih]

theorem list_filter_sum (xs : List Nat) (p : Nat → Bool) (f : Nat → ℝ) :
    ((xs.filter p).map f).sum=(xs.map fun n=>if p n then f n else 0).sum := by
  induction xs with
  | nil=>rfl
  | cons x xs ih=>
    cases hx : p x <;> simp [hx,ih]

theorem list_flatMap_sum (xs : List Nat) (g : Nat → List Nat) (f : Nat → ℝ) :
    ((xs.flatMap g).map f).sum=(xs.map fun n=>((g n).map f).sum).sum := by
  induction xs with
  | nil=>rfl
  | cons x xs ih=>simp [ih]

/-- Exact equality to the independent occurrence-list support; duplicate keys
retain multiplicity, and all count-selected blocks and columns occur. -/
theorem total_support (c : Config) (s : State R) (i : Fin c.BM) (f : Nat → ℝ) :
    total c s i f=((VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.support c.BN (sequenceLength c s) (rowStart c s+i.val)
      (blockCount c s) (columnCount c s) (blockOffset c s) (columnIndex c s)).map f).sum := by
  unfold total VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.support
  rw [List.map_append,List.sum_append,list_flatMap_sum]
  simp only [List.map_map,list_range_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro b _
  rw [list_filter_sum]
  simp only [List.map_map,list_range_sum,Function.comp_apply,decide_eq_true_eq,blockSum]
  simpa only [Function.comp_apply] using Fin.sum_univ_eq_sum_range
    (fun j=>if blockOffset c s b+j<sequenceLength c s ∧ blockOffset c s b+j≤rowStart c s+i.val
      then f (blockOffset c s b+j) else 0) c.BN

def expected (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (i : Fin c.BM) (d : Fin c.BD) : ℝ :=
  VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.expected c.BD c.BN (sequenceLength c s) (rowStart c s+i.val)
    (blockCount c s) (columnCount c s) c.scale (qAt c s μ (rowStart c s+i.val)) (kAt c s μ) (vAt c s μ)
    (blockOffset c s) (columnIndex c s) d.val

theorem total_ratio (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (i : Fin c.BM) (d : Fin c.BD) :
    total c s i (fun j=>pow2 (logicalScore c s μ i j)*vAt c s μ j d.val)/
      total c s i (fun j=>pow2 (logicalScore c s μ i j))=expected c s μ i d := by
  rw [total_support,total_support]
  have hscore : ∀ j,logicalScore c s μ i j=VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.score c.BD c.scale
      (qAt c s μ (rowStart c s+i.val)) (kAt c s μ j) := by
    intro j
    unfold logicalScore VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.score
    rw [Fin.sum_univ_eq_sum_range (fun d=>qAt c s μ (rowStart c s+i.val) d*kAt c s μ j d) c.BD]
  simp only [expected,VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.expected,hscore]

def answer (c : Config) (s : State R) (i : Fin c.BM) (d : Fin c.BD) : R :=
  let a:=finalValues algebra dotModel c s
  if dotModel.equal (a.normalizer (i,())) (some 0) then some 0
  else WithBot.realDiv (a.output (i,d,())) (a.normalizer (i,()))

/-- The repaired count-driven recurrence returns independent sparse attention,
including empty support and wholly masked prefixes. -/
theorem answer_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (i : Fin c.BM) (d : Fin c.BD) (hi : rowStart c s+i.val<sequenceLength c s) (hbn : 0<c.BN) :
    answer c s i d=some (expected c s μ i d) := by
  obtain ⟨result,⟨hm,hl,ht⟩,⟨hL,hT,_⟩⟩:=final_consistent c s μ h i d hi hbn
  dsimp only [answer]
  rw [hl,ht]
  have ratio : result.numerator/result.denominator=expected c s μ i d := by
    rw [hL,hT,mul_div_mul_left _ _ (ne_of_gt (pow2_pos _)),total_ratio]
  by_cases hz : result.denominator=0
  · have he : expected c s μ i d=0 := by rw [←ratio,hz,div_zero]
    simp [dotModel,hz,he]
  · simp only [dotModel,decide_eq_true_eq]
    split
    · rename_i hzero
      exact (hz (Option.some.inj hzero)).elim
    · change some (result.numerator/result.denominator)=_
      rw [ratio]

#axiomsClean final_consistent
#axiomsClean total_support
#axiomsClean answer_real
#specNonCircular expected avoiding [FP.Structural.exec,Source.finalValues,Source.update]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
