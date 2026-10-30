import bench.tritonbench_g.sgmv_expand_slice.Proofs.RealBackend
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceMemory
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FoldValues

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 20000

theorem update_formula (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (s : State ℝ) (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ)
    (memory : SourceMemory.Inputs abi p shape s X W) (c : SourceBody.Coordinates)
    (rows : ∀ r, c.start+c.rows r < shape.tokens) (adapter : c.adapter < shape.adapters)
    (columns : ∀ j, c.columns j < shape.columns) (block : Nat)
    (hb : block < (p.inner+15)/16) (acc : Values ℝ .real [32,32]) (i : TileIndex [32,32]) :
    SourceBody.update (algebra rest) dot abi p c s block acc i = acc i +
      ∑ k : Fin 16, if block*16+k.val < shape.inner then
        X (c.start+c.rows i.1) (block*16+k.val)*W c.adapter (c.columns i.2.1) (block*16+k.val) else 0 := by
  unfold SourceBody.update
  rw [selected_dot]
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  rw [SourceMemory.a_values _ _ _ _ _ _ _ memory c rows block hb,
    SourceMemory.b_values _ _ _ _ _ _ _ memory c adapter columns block hb]
  simp only [algebra]
  split_ifs <;> simp

/-- Source K blocks partition the padded finite range. The induction uses the
actual ordered update at each block and removes only the explicit zero tail. -/
theorem accumulated_contraction (rest : Algebra ℝ) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (s : State ℝ) (X : Nat → Nat → ℝ) (W : Nat → Nat → Nat → ℝ)
    (memory : SourceMemory.Inputs abi p shape s X W) (c : SourceBody.Coordinates)
    (rows : ∀ r, c.start+c.rows r < shape.tokens) (adapter : c.adapter < shape.adapters)
    (columns : ∀ j, c.columns j < shape.columns) (i : TileIndex [32,32]) :
    SourceBody.accumulated (algebra rest) dot abi p c s 0 (SourceInit.zero (algebra rest)) i =
      contraction shape X W c.adapter (c.start+c.rows i.1) (c.columns i.2.1) := by
  let f : Nat → ℝ := fun k => if k < shape.inner then
    X (c.start+c.rows i.1) k * W c.adapter (c.columns i.2.1) k else 0
  have hprefix (n : Nat) (hn : n ≤ (p.inner+15)/16) :
      ((List.range n).foldl (fun acc block => SourceBody.update (algebra rest) dot abi p c s block acc)
        (SourceInit.zero (algebra rest))) i = ∑ k ∈ Finset.range (n*16), f k := by
    induction n with
    | zero => simp [SourceInit.zero,algebra]
    | succ n ih =>
        rw [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil]
        rw [update_formula rest abi p shape s X W memory c rows adapter columns n (by omega),
          ih (by omega),Nat.succ_mul,Finset.sum_range_add]
        congr 1
  unfold SourceBody.accumulated
  rw [foldRange_range,hprefix _ (Nat.le_refl _)]
  have hb : shape.inner ≤ ((p.inner+15)/16)*16 := by rw [memory.inner]; omega
  have ht : (∑ k ∈ Finset.range (((p.inner+15)/16)*16),f k) =
      ∑ k ∈ Finset.range shape.inner,f k := by
    symm
    apply Finset.sum_subset (Finset.range_mono hb)
    intro k _ hk
    simp only [Finset.mem_range] at hk
    simp [f,hk]
  rw [ht]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [f,if_pos (Finset.mem_range.mp hk)]

#axiomsClean update_formula
#axiomsClean accumulated_contraction
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
