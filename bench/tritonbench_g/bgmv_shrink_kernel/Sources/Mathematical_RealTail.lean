import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealSetup

/-! The complete scaling and output suffix. Ordinary store overwrites a cell;
atomic addition adds this program's contribution to that cell's initial value.
These are real per-program semantics, with no launch interleaving claim. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTail
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def offset (p : Parameters) (batch n : Nat) : Nat :=
  batch * p.outputBatchStride + n * p.outputStride

private def scaled (p : Parameters) (s : BlockState) (g : Fin p.BN → ℝ) : BlockState :=
  s.setReg "accumulator" .real [p.BN] ⟨fun i => some (g i.1 * p.scaling)⟩

private def offsets (p : Parameters) (s : BlockState) (g : Fin p.BN → ℝ) : BlockState :=
  (scaled p s g).setReg "offset_cn" .nat [p.BN] (Tile.vec (fun i => i.val))

private def pointers (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) : BlockState :=
  (offsets p s g).setReg "c_ptr" .ptr [p.BN] ⟨fun i => (p.output, offset p batch i.1.val)⟩

def prepared (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) : BlockState :=
  (pointers p s batch g).setReg "c_mask" .bool [p.BN] ⟨fun i => decide (i.1.val < p.N)⟩

def scatter (p : Parameters) (s : BlockState) (batch : Nat) (v : Fin p.BN → ℝ) : BlockState :=
  (TileShape.allIndices [p.BN]).foldl (fun acc i =>
    if decide (i.1.val < p.N) then
      acc.writeMem p.output (offset p batch i.1.val)
        (if p.splits = 1 then v i.1 else acc.readMem p.output (offset p batch i.1.val) + v i.1)
    else acc) s

def result (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) : BlockState :=
  scatter p (prepared p s batch g) batch (fun n => g n * p.scaling)

theorem scatter_active (p : Parameters) (s : BlockState) (batch : Nat) (v : Fin p.BN → ℝ)
    (hstride : 0 < p.outputStride) (n : Fin p.BN) (hn : n.val < p.N) :
    (scatter p s batch v).readMem p.output (offset p batch n.val) =
      if p.splits = 1 then v n else s.readMem p.output (offset p batch n.val) + v n := by
  have hu : ∀ b ∈ TileShape.allIndices [p.BN], decide (b.1.val < p.N) = Bool.true →
      offset p batch b.1.val = offset p batch n.val → b = (n, PUnit.unit) := by
    intro b _ _ he
    have hv : b.1.val = n.val :=
      Nat.eq_of_mul_eq_mul_right hstride (Nat.add_left_cancel he)
    exact Prod.ext (Fin.ext hv) rfl
  unfold scatter
  by_cases hs : p.splits = 1
  · simp only [hs, if_true]
    exact foldl_store_at _ _ _ _ _ s (n, PUnit.unit)
      (TileShape.mem_allIndices _ _) (by simpa) rfl hu (TileShape.allIndices_nodup _)
  · simp only [hs, if_false]
    exact foldl_atomicAdd_at _ _ _ _ _ s (n, PUnit.unit)
      (TileShape.mem_allIndices _ _) (by simpa) rfl hu (TileShape.allIndices_nodup _)

/-- Every untouched cell is preserved as a typed cell, including metadata,
masked output lanes and other regions. No alias restriction is needed here. -/
theorem scatter_frame (p : Parameters) (s : BlockState) (batch : Nat) (v : Fin p.BN → ℝ)
    (r : RegionName) (o : Nat)
    (h : ∀ n : Fin p.BN, n.val < p.N → ¬(p.output = r ∧ offset p batch n.val = o)) :
    (scatter p s batch v).mem r o = s.mem r o := by
  unfold scatter
  generalize TileShape.allIndices [p.BN] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    by_cases hi : i.1.val < p.N
    · have hmiss : ¬ (r = p.output ∧ o = offset p batch i.1.val) :=
        fun he => h i.1 hi ⟨he.1.symm, he.2.symm⟩
      simp only [hi, decide_true, if_true, BlockState.writeMem_mem, if_neg hmiss]
    · simp [hi]

@[simp] theorem prepared_mem (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) :
    (prepared p s batch g).mem = s.mem := rfl

theorem tail_readback (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ)
    (hstride : 0 < p.outputStride) (n : Fin p.BN) (hn : n.val < p.N) :
    (result p s batch g).readMem p.output (offset p batch n.val) =
      if p.splits = 1 then g n * p.scaling
      else s.readMem p.output (offset p batch n.val) + g n * p.scaling := by
  exact scatter_active p (prepared p s batch g) batch _ hstride n hn

theorem tail_frame (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ)
    (r : RegionName) (o : Nat)
    (h : ∀ n : Fin p.BN, n.val < p.N → ¬(p.output = r ∧ offset p batch n.val = o)) :
    (result p s batch g).mem r o = s.mem r o :=
  scatter_frame p (prepared p s batch g) batch _ r o h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTail
