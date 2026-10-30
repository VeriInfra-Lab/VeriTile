import bench.tritonbench_g.cache_transform_triton.Proofs.Execution
import bench.tritonbench_g.cache_transform_triton.Proofs.Bounds

namespace VeriTile.Bench.Repairs.CacheTransform
open Lean VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 100000
set_option linter.unusedSimpArgs false

def tileParameters (p : Parameters) (k : Nat) : Parameters :=
  Parameters.mk p.cache p.rows p.output p.count p.rowBytes (chunkStart k) 256

def tileState {α : Type} (s : State α) (k : Nat) : State α :=
  { s with pids := fun ax => if ax = 0 then localPid k else 0 }

theorem tile_count (p : Parameters) (k : Nat) : (tileParameters p k).count = p.count := rfl
theorem tile_width (p : Parameters) (k : Nat) : (tileParameters p k).rowBytes = p.rowBytes := rfl
theorem tile_output (p : Parameters) (k : Nat) : (tileParameters p k).output = p.output := rfl
theorem tile_cache (p : Parameters) (k : Nat) : (tileParameters p k).cache = p.cache := rfl
theorem tile_rows (p : Parameters) (k : Nat) : (tileParameters p k).rows = p.rows := rfl
theorem tile_mem {α : Type} (s : State α) (k : Nat) : (tileState s k).mem = s.mem := rfl

def tile {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (k : Nat) (s : State α) : State α :=
  (execution M (tileParameters p k) (tileState s k)).val

theorem tile_success {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (k : Nat) (s : State α) :
    FP.Structural.exec M (source (tileParameters p k)) (tileState s k) =
      some (tile M p k s) := (execution M _ _).property

theorem tile_index {α : Type} (p : Parameters) (s : State α) (k : Nat)
    (i : TileIndex [256]) :
    index (tileParameters p k) (tileState s k) i = k * 256 + i.1.val := by
  have hp : localPid k < 18446744073709551616 := lt_trans (localPid_bound k) (by decide)
  have hi : i.1.val < 18446744073709551616 := lt_trans i.1.isLt (by decide)
  simp only [index_eq, tileParameters, tileState, ↓reduceIte, Nat.mod_eq_of_lt hp,
    Nat.mod_eq_of_lt hi]
  have := chunk_identity k
  omega

theorem lane_owner (k lane : Nat) (h : lane < 256) : (k * 256 + lane) / 256 = k := by omega

theorem tile_miss {α : Type} (p : Parameters) (s : State α)
    (k : Nat) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ o ≥ p.count ∨ o / 256 ≠ k) :
    r ≠ (tileParameters p k).output ∨ ∀ i, (index (tileParameters p k) (tileState s k) i < (tileParameters p k).count) →
      o ≠ index (tileParameters p k) (tileState s k) i := by
  rw [tile_output, tile_count]
  rcases h with h | h
  · exact Or.inl h
  · right
    intro i hi
    exact index_miss (index (tileParameters p k) (tileState s k)) p.count o k
      (tile_index p s k) h i hi

theorem tile_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (k : Nat) (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ o ≥ p.count ∨ o / 256 ≠ k) :
    (tile M p k s).mem r o = s.mem r o :=
  frame M (tileParameters p k) (tileState s k) r o (tile_miss p s k r o h)

/-- Ordinary direct-view/metadata read contract, only on allocated windows.
The cache and row table may alias each other; the fresh output aliases neither. -/
structure GatherInput {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (n capacity : Nat) (rows cache : Nat → Nat) : Prop where
  shape : p.count = n * p.rowBytes
  width : 0 < p.rowBytes
  distinctCache : p.cache ≠ p.output
  distinctRows : p.rows ≠ p.output
  table : ∀ q, q < n → s.mem p.rows q = .mk .int (rows q : Int)
  rowBound : ∀ q, q < n → rows q < capacity
  capacityBound : capacity ≤ 9223372036854775807
  payload : ∀ o, o < capacity * p.rowBytes → s.mem p.cache o = .mk .nat (cache o)

def byteTarget (p : Parameters) (rows cache : Nat → Nat) (o : Nat) : Nat :=
  cache (rows (o / p.rowBytes) * p.rowBytes + o % p.rowBytes)

theorem tile_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (k : Nat) (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (input : GatherInput p s n capacity rows cache)
    (o : Nat) (ho : o < p.count) (hk : o / 256 = k) :
    (tile M p k s).mem p.output o = .mk .nat (byteTarget p rows cache o) := by
  let i : TileIndex [256] := (⟨o % 256, Nat.mod_lt _ (by decide)⟩, PUnit.unit)
  have hidx : index (tileParameters p k) (tileState s k) i = o := by
    exact (tile_index p s k i).trans (by
      change k * 256 + o % 256 = o
      rw [← hk]
      simpa only [Nat.mul_comm] using Nat.div_add_mod o 256)
  have hd : o / p.rowBytes < n :=
    (Nat.div_lt_iff_lt_mul input.width).mpr (input.shape ▸ ho)
  have hnat : rows (o / p.rowBytes) < 18446744073709551616 :=
    lt_trans (lt_of_lt_of_le (input.rowBound _ hd) input.capacityBound) (by decide)
  have hr : (rows (o / p.rowBytes) : Int) < 18446744073709551616 := by
    exact_mod_cast hnat
  have hoff : inputOffset (tileParameters p k) (tileState s k) i =
      rows (o / p.rowBytes) * p.rowBytes + o % p.rowBytes :=
    inputOffset_at (tileParameters p k) (tileState s k) i o (rows (o / p.rowBytes)) hidx
      ho hnat (input.table _ hd)
  have hb : rows (o / p.rowBytes) * p.rowBytes + o % p.rowBytes < capacity * p.rowBytes :=
    (gather_bounds n p.rowBytes capacity o rows input.width (input.shape ▸ Nat.lt_of_le_of_lt (Nat.zero_le o) ho)
      (input.shape ▸ ho) input.rowBound).2.2
  apply readback_at M (tileParameters p k) (tileState s k) i o _ hidx ho
    (by change 256 ≤ 18446744073709551616; decide)
  change Cell.mk .nat ((s.mem p.cache (inputOffset (tileParameters p k) (tileState s k) i)).read .nat) = _
  rw [hoff, input.payload _ hb, Cell.read_same]
  rfl

def runTiles {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) : Nat → State α
  | 0 => s
  | k + 1 => tile M p k (runTiles M p s k)

theorem run_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n : Nat) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ o ≥ p.count) :
    (runTiles M p s n).mem r o = s.mem r o := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [runTiles, tile_frame M p n _ r o (h.imp_right Or.inl), ih]

theorem run_input {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (h : GatherInput p s n capacity rows cache) (k : Nat) :
    GatherInput p (runTiles M p s k) n capacity rows cache := by
  refine { h with table := ?_, payload := ?_ }
  · intro q hq
    rw [run_frame M p s k p.rows q (Or.inl h.distinctRows)]
    exact h.table q hq
  · intro q hq
    rw [run_frame M p s k p.cache q (Or.inl h.distinctCache)]
    exact h.payload q hq

theorem run_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (h : GatherInput p s n capacity rows cache) (k o : Nat)
    (ho : o < p.count) (hk : o / 256 < k) :
    (runTiles M p s k).mem p.output o = .mk .nat (byteTarget p rows cache o) := by
  induction k with
  | zero => omega
  | succ k ih =>
    rw [runTiles]
    by_cases heq : o / 256 = k
    · exact tile_readback M p k _ n capacity rows cache (run_input M p s n capacity rows cache h k) o ho heq
    · rw [tile_frame M p k _ p.output o (Or.inr (Or.inr heq))]
      exact ih (by omega)

def gather {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  runTiles M p s (blocks p.count)

/-- All bytes of every output row, both tails and every host grid chunk.
This is a fold of successful executions of the actual source, not an ideal loop. -/
theorem gather_correct {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (h : GatherInput p s n capacity rows cache) :
    (∀ o, o < p.count → (gather M p s).mem p.output o = .mk .nat (byteTarget p rows cache o)) ∧
    (∀ r o, r ≠ p.output ∨ o ≥ p.count → (gather M p s).mem r o = s.mem r o) := by
  constructor
  · intro o ho
    exact run_readback M p s n capacity rows cache h _ o ho (block_coverage _ _ ho).1
  · exact run_frame M p s _

theorem gather_correct_total {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (h : 0 < p.count → GatherInput p s n capacity rows cache) :
    (∀ o, o < p.count → (gather M p s).mem p.output o = .mk .nat (byteTarget p rows cache o)) ∧
    (∀ r o, r ≠ p.output ∨ o ≥ p.count → (gather M p s).mem r o = s.mem r o) := by
  by_cases hpos : 0 < p.count
  · exact gather_correct M p s n capacity rows cache (h hpos)
  · have hz : p.count = 0 := by omega
    constructor
    · intro o ho
      omega
    · intro r o _
      simp [gather, blocks, hz, runTiles]

theorem tile_output_disjoint {α : Type} (p : Parameters) (s t : State α)
    (k j : Nat) (h : k ≠ j) (a b : TileIndex [256]) :
    index (tileParameters p k) (tileState s k) a ≠
      index (tileParameters p j) (tileState t j) b := by
  rw [tile_index, tile_index]
  have := a.1.isLt
  have := b.1.isLt
  omega

#axiomsClean tile_success
#axiomsClean tile_readback
#axiomsClean gather_correct
#axiomsClean gather_correct_total
end
end VeriTile.Bench.Repairs.CacheTransform
