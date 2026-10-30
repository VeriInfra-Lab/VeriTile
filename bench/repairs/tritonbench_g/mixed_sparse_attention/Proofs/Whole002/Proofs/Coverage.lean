import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.RealUpdate
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def prefixSum (N t : Nat) (f : Nat → ℝ) : ℝ := ∑ j∈Finset.range t,if j<N then f j else 0

def chunk (BN N start : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j : Fin BN,if start+j.val<N then f (start+j.val) else 0

theorem prefix_step (BN N start : Nat) (f : Nat → ℝ) :
    prefixSum N (start+BN) f=prefixSum N start f+chunk BN N start f := by
  simp only [prefixSum,chunk,Finset.sum_range_add]
  rw [Fin.sum_univ_eq_sum_range (fun j=>if start+j<N then f (start+j) else 0) BN]

theorem prefix_done (N start : Nat) (f : Nat → ℝ) (h : N≤start) :
    prefixSum N start f=∑ j∈Finset.range N,f j := by
  rw [show start=N+(start-N) by omega,prefixSum,Finset.sum_range_add]
  have first : (∑ j∈Finset.range N,if j<N then f j else 0)=∑ j∈Finset.range N,f j := by
    apply Finset.sum_congr rfl
    intro j hj
    simp [Finset.mem_range.mp hj]
  simpa [first]

def blockOffset (c : Config) (s : State R) (b : Nat) : Nat :=
  (s.mem (Region.cast c.offsets) (metadataIndex c s*c.NS+b)).read .nat
def columnIndex (c : Config) (s : State R) (j : Nat) : Nat :=
  (s.mem (Region.cast c.columns) (metadataIndex c s*c.NV+j)).read .nat

def blockSum (c : Config) (s : State R) (i : Fin c.BM) (f : Nat → ℝ) (b : Nat) : ℝ :=
  ∑ j : Fin c.BN,if blockOffset c s b+j.val<sequenceLength c s ∧
    blockOffset c s b+j.val≤rowStart c s+i.val then f (blockOffset c s b+j.val) else 0

def tileSum (c : Config) (s : State R) (i : Fin c.BM) (ph : Bool) (start : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j : Fin c.BN,if selected c s ph start i j then f (columnValues c s ph start j) else 0

theorem block_sum (c : Config) (s : State R) (i : Fin c.BM) (start : Nat) (f : Nat → ℝ)
    (hs : start<blockCount c s) :
    tileSum c s i Bool.false start f=blockSum c s i f start := by
  simp [tileSum,blockSum,selected,keyActive,columnValues,blockOffset,hs]

theorem column_sum (c : Config) (s : State R) (i : Fin c.BM) (start : Nat) (f : Nat → ℝ)
    (hs : start<columnCount c s) :
    tileSum c s i Bool.true start f=chunk c.BN (columnCount c s) start (fun j=>f (columnIndex c s j)) := by
  unfold tileSum chunk
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : start+j.val<columnCount c s
  · simp [selected,keyActive,columnValues,columnIndex,hs,hj,Nat.add_assoc]
  · simp [selected,keyActive,hs,hj]

theorem tile_denominator (c : Config) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : Fin c.BM) (d : Fin c.BD) (ph : Bool) (start : Nat) :
    VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.denominator (tile c s μ ph start i d)=tileSum c s i ph start (fun j=>pow2 (logicalScore c s μ i j)) := by
  simp [VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.denominator,tile,tileSum]

theorem tile_numerator (c : Config) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : Fin c.BM) (d : Fin c.BD) (ph : Bool) (start : Nat) :
    VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.numerator (tile c s μ ph start i d)=tileSum c s i ph start
      (fun j=>pow2 (logicalScore c s μ i j)*vAt c s μ j d.val) := by
  simp [VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.numerator,tile,tileSum]

/-- General transport through the exact dynamic range, retaining count,
stride and multiplicity. The only step premise is the proved local tile sum. -/
theorem loop_consistent (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (i : Fin c.BM) (d : Fin c.BD) (hi : rowStart c s+i.val<sequenceLength c s)
    (ph : Bool) (stop stride : Nat) (hstride : 0<stride)
    (mass value : Nat → ℝ)
    (hmass : ∀ start,start<stop → VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.denominator (tile c s μ ph start i d)=chunk stride stop start mass)
    (hvalue : ∀ start,start<stop → VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.numerator (tile c s μ ph start i d)=chunk stride stop start value)
    (L T : ℝ) (start : Nat) (a : Accumulator R c.BM c.BD) (b : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Acc)
    (hab : RowAgrees c i d a b)
    (hc : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Consistent b (L+prefixSum stop start mass) (T+prefixSum stop start value)) :
    ∃ result,RowAgrees c i d (foldRange (update algebra dotModel c s ph) start stop stride a) result ∧
      VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Consistent result (L+∑ j∈Finset.range stop,mass j) (T+∑ j∈Finset.range stop,value j) := by
  by_cases hs : start<stop
  · have hstep:=VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.step_consistent b (tile c s μ ph start i d) _ _ hc
    rw [hmass start hs,hvalue start hs] at hstep
    have hnext : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Consistent (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.step b (tile c s μ ph start i d))
        (L+prefixSum stop (start+stride) mass) (T+prefixSum stop (start+stride) value) := by
      simpa [prefix_step,add_assoc] using hstep
    obtain ⟨result,hr,hc⟩:=loop_consistent c s μ h i d hi ph stop stride hstride mass value hmass hvalue L T
      (start+stride) (update algebra dotModel c s ph start a) (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.step b (tile c s μ ph start i d))
      (update_agrees c s μ h ph start i d a b hi hab) hnext
    refine ⟨result,?_,hc⟩
    rw [foldRange]
    simpa only [show ¬stride=0 by omega,hs,if_false,if_true] using hr
  · refine ⟨b,?_,?_⟩
    · rw [foldRange]
      simpa only [show ¬stride=0 by omega,hs,if_false] using hab
    · simpa [prefix_done stop start mass (by omega),prefix_done stop start value (by omega)] using hc
termination_by stop-start
decreasing_by omega

#axiomsClean prefix_step
#axiomsClean prefix_done
#axiomsClean block_sum
#axiomsClean column_sum
#axiomsClean loop_consistent
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
