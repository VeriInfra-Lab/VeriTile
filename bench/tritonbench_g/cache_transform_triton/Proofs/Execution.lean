import bench.tritonbench_g.cache_transform_triton.Kernels
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.CacheTransform
open Lean VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Keep the native index behind a checked equation to avoid eager kernel
reduction of large machine-width remainders in dependent mask types. -/
private theorem index_exists {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) : ∃ n : Nat,
    n = s.pids 0 % 18446744073709551616 * p.block + i.1.val % 18446744073709551616 + p.start := ⟨_, rfl⟩

def index {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.block]) : Nat :=
  Classical.choose (index_exists p s i)

theorem index_eq {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.block]) :
    index p s i = s.pids 0 % 18446744073709551616 * p.block + i.1.val % 18446744073709551616 + p.start :=
  Classical.choose_spec (index_exists p s i)

def safe {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.block]) : Nat :=
  if (index p s i < p.count) then index p s i else 0

def inputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) : Nat :=
  (((s.mem p.rows (safe p s i / p.rowBytes)).read .int) % 18446744073709551616).toNat * p.rowBytes +
    safe p s i % p.rowBytes

def copied {α : Type} [Inhabited α] (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i =>
    if (index p s i < p.count) then t.write p.output (safe p s i)
      (.mk .nat ((s.mem p.cache (inputOffset p s i)).read .nat)) else t) s

/-- Every actual source statement executes for any structural backend.
There are no floating callbacks, no numerical laws and no output premise. -/
private def rawExecution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, ComputeDType.eraseDType, Region.cast, Option.bind]
  rfl

private theorem rawExecution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (rawExecution M p s).val.mem = (copied p s).mem := by
  simp [rawExecution, copied, masked_scatter_memory, index_eq, safe, inputOffset,
    Broadcast.leftIndex, Broadcast.rightIndex, integralFloorDiv, integralMod,
    IntegerExpr.sameShape, FP.Structural.bop]
  congr 1
  funext mem i
  by_cases hi : index p s i < p.count
  · simp only [index_eq] at hi
    simp only [if_pos hi]
  · simp only [index_eq] at hi
    simp only [if_neg hi]

/-- Seal the checked witness behind an existential proof. This avoids expanding
the entire source evaluator during downstream kernel type checking. -/
theorem execution_exists {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (source p) s = some t ∧ t.mem = (copied p s).mem :=
  ⟨(rawExecution M p s).val, (rawExecution M p s).property, rawExecution_memory M p s⟩

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} :=
  ⟨Classical.choose (execution_exists M p s), (Classical.choose_spec (execution_exists M p s)).1⟩

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (copied p s).mem :=
  (Classical.choose_spec (execution_exists M p s)).2

theorem output_injective {α : Type} (p : Parameters) (s : State α)
    (hblock : p.block ≤ 18446744073709551616) :
    ∀ a b, (index p s a < p.count) → (index p s b < p.count) → safe p s a = safe p s b → a = b := by
  intro a b ha hb h
  unfold safe at h
  rw [if_pos ha, if_pos hb] at h
  simp only [index_eq] at h
  rw [Nat.mod_eq_of_lt (lt_of_lt_of_le a.1.isLt hblock),
    Nat.mod_eq_of_lt (lt_of_lt_of_le b.1.isLt hblock)] at h
  exact Prod.ext (Fin.ext (Nat.add_left_cancel (Nat.add_right_cancel h))) (Subsingleton.elim _ _)

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (i : TileIndex [p.block]) (hi : (index p s i < p.count))
    (hblock : p.block ≤ 18446744073709551616) :
    (execution M p s).val.mem p.output (index p s i) =
      .mk .nat ((s.mem p.cache (inputOffset p s i)).read .nat) := by
  rw [execution_memory]
  have h := masked_readback_active p.output (safe p s)
    (fun j => Cell.mk .nat ((s.mem p.cache (inputOffset p s j)).read .nat))
    (fun i => index p s i < p.count) s i hi (output_injective p s hblock)
  simpa [copied, safe, hi] using h

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ i, (index p s i < p.count) → o ≠ index p s i) :
    (execution M p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  unfold copied
  rw [State.masked_scatter_eq_filter]
  apply State.scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · right
    intro i hi
    have ha : (index p s i < p.count) := by simpa using (List.mem_filter.mp hi).2
    simpa [safe, ha] using h i ha

theorem readback_at {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (i : TileIndex [p.block]) (o : Nat) (value : Cell α)
    (hindex : index p s i = o) (ho : o < p.count)
    (hblock : p.block ≤ 18446744073709551616)
    (hvalue : Cell.mk .nat ((s.mem p.cache (inputOffset p s i)).read .nat) = value) :
    (execution M p s).val.mem p.output o = value := by
  have hi : (index p s i < p.count) := by
    rw [hindex]
    exact ho
  rw [← hindex]
  exact (readback M p s i hi hblock).trans hvalue

theorem inputOffset_at {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) (o row : Nat) (hindex : index p s i = o)
    (ho : o < p.count) (hrow : row < 18446744073709551616)
    (hm : s.mem p.rows (o / p.rowBytes) = .mk .int (row : Int)) :
    inputOffset p s i = row * p.rowBytes + o % p.rowBytes := by
  have ha : (index p s i < p.count) := by
    rw [hindex]
    exact ho
  unfold inputOffset
  have hs : safe p s i = o := by
    unfold safe
    rw [if_pos ha]
    exact hindex
  rw [hs, hm, Cell.read_same]
  rw [Int.emod_eq_of_lt (Int.natCast_nonneg _) (by exact_mod_cast hrow), Int.toNat_natCast]

theorem index_miss (idx : TileIndex [256] → Nat) (count o k : Nat)
    (hindex : ∀ i, idx i = k * 256 + i.1.val)
    (h : o ≥ count ∨ o / 256 ≠ k) :
    ∀ i, idx i < count → o ≠ idx i := by
  intro i hi heq
  rcases h with h | h
  · exact Nat.not_lt_of_ge h (heq.symm ▸ hi)
  · apply h
    have heq' := heq.trans (hindex i)
    have hl := i.1.isLt
    omega

attribute [irreducible] execution

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.CacheTransform
