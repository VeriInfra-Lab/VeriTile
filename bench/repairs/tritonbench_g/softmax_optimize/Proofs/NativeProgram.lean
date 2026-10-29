import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Lowering
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.NativeStores

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.NativeProgram
open Bytes Graph Lowering
set_option maxHeartbeats 1000000

/-- Actual tile/lane visitation, retaining full and final masked loops. -/
def tiles (count B : Nat) : List Nat :=
  (List.range count).flatMap (fun tile => (List.range B).map (fun lane => tile * B + lane))

theorem tiles_range (count B : Nat) : tiles count B = List.range (count * B) := by
  induction count with
  | zero => simp [tiles]
  | succ count ih =>
    simp only [tiles, List.range_succ, List.flatMap_append, List.flatMap_singleton] at *
    rw [ih, Nat.succ_mul, List.range_add]

def columns (N B : Nat) : List Nat :=
  (tiles (Config.fullTiles N B) B ++
    (List.range B).map (fun lane => Config.fullTiles N B * B + lane)).filter (fun j => j < N)

theorem filter_range (N count : Nat) :
    (List.range count).filter (fun j => j < N) = List.range (min N count) := by
  induction count with
  | zero => simp
  | succ count ih =>
    rw [List.range_succ, List.filter_append, ih]
    by_cases h : count < N
    · have hc : min N count = count := min_eq_right (by omega)
      have hn : min N (count + 1) = count + 1 := min_eq_right (by omega)
      simp [h, hc, hn, List.range_succ]
    · have hc : min N count = N := min_eq_left (by omega)
      have hn : min N (count + 1) = N := min_eq_left (by omega)
      simp [h, hc, hn]

theorem columns_range (N B : Nat) (hN : 0 < N) (hB : 0 < B) :
    columns N B = List.range N := by
  rw [columns, tiles_range, ← List.range_add, filter_range]
  have cover := Config.tail_covers hN hB
  rw [min_eq_left (by simpa [Nat.add_mul] using cover)]

/-- Filtering full-loop lanes changes nothing: their source loads and stores
are unmasked. Only the tail actually consumes a mask. -/
theorem full_active (N B tile lane : Nat) (hN : 0 < N)
    (ht : tile < Config.fullTiles N B) (hl : lane < B) : tile * B + lane < N :=
  Config.full_tile_active hN ht hl

def gridIndices (M N B : Nat) : List Nat :=
  (List.range M).flatMap (fun row => (columns N B).map (fun column => row * N + column))

theorem grid_range (M N B : Nat) (hN : 0 < N) (hB : 0 < B) :
    gridIndices M N B = List.range (M * N) := by
  simp only [gridIndices, columns_range N B hN hB]
  exact tiles_range M N

theorem unique_coverage (M N B : Nat) (hN : 0 < N) (hB : 0 < B) :
    (gridIndices M N B).Nodup ∧
    ∀ index, index ∈ gridIndices M N B ↔ index < M * N := by
  rw [grid_range M N B hN hB]
  exact ⟨List.nodup_range, fun _ => List.mem_range⟩

def rowInput (heap : Heap) (inputBase N row : Nat) : Nat → Word :=
  fun column => if column < N then loadWord heap (inputBase + 4 * (row * N + column)) else 0

def word (ops : Graph.Operations Word) (compiler : Compiler B) (hB : 0 < B)
    (heap : Heap) (inputBase N row column : Nat) : Word :=
  Graph.eval ops (rowInput heap inputBase N row)
    (outputExpr compiler N column ⟨column % B, Nat.mod_lt _ hB⟩)

def run (ops : Graph.Operations Word) (compiler : Compiler B) (hB : 0 < B)
    (heap : Heap) (inputBase outputBase M N : Nat) : Heap :=
  NativeStores.run heap outputBase
    (fun index => word ops compiler hB heap inputBase N (index / N) (index % N))
    (gridIndices M N B)

/-- Complete actual word-store execution of the locally lowered typed body.
No successful-execution or output premise occurs; arbitrary total FP32 primitive
operations and all bit patterns are supported. -/
theorem stored_output (ops : Graph.Operations Word) (compiler : Compiler B) (hB : 0 < B)
    (heap : Heap) (inputBase outputBase M N : Nat) (hN : 0 < N)
    (row : Fin M) (column : Fin N) :
    loadWord (run ops compiler hB heap inputBase outputBase M N)
      (outputBase + 4 * (row.val * N + column.val)) =
      Graph.eval ops (rowInput heap inputBase N row)
        (RowGraph.output compiler.profile N column ⟨column.val % B, Nat.mod_lt _ hB⟩) := by
  unfold run
  rw [grid_range M N B hN hB]
  have hi : row.val * N + column.val < M * N := by
    have hr := row.isLt
    have hc := column.isLt
    nlinarith
  rw [NativeStores.readback _ _ _ _ _ (List.mem_range.mpr hi)]
  have hd : (row.val * N + column.val) / N = row.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt column.isLt, Nat.zero_add]
  have hm : (row.val * N + column.val) % N = column.val := by
    simp [Nat.add_mod, Nat.mod_eq_of_lt column.isLt]
  simp only [hd, hm, word, output_graph]

theorem stored_byte (ops : Graph.Operations Word) (compiler : Compiler B) (hB : 0 < B)
    (heap : Heap) (inputBase outputBase M N : Nat) (hN : 0 < N)
    (row : Fin M) (column : Fin N) (byte : Fin 4) :
    run ops compiler hB heap inputBase outputBase M N
      (outputBase + 4 * (row.val * N + column.val) + byte.val) = encode (
      Graph.eval ops (rowInput heap inputBase N row)
        (RowGraph.output compiler.profile N column ⟨column.val % B, Nat.mod_lt _ hB⟩)) byte := by
  unfold run
  rw [grid_range M N B hN hB]
  have hi : row.val * N + column.val < M * N := by
    have hr := row.isLt
    have hc := column.isLt
    nlinarith
  rw [NativeStores.byte,if_pos (List.mem_range.mpr hi)]
  have hd : (row.val * N + column.val) / N = row.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt column.isLt, Nat.zero_add]
  have hm : (row.val * N + column.val) % N = column.val := by
    simp [Nat.add_mod, Nat.mod_eq_of_lt column.isLt]
  simp only [hd, hm, word, output_graph]

theorem full_frame (ops : Graph.Operations Word) (compiler : Compiler B) (hB : 0 < B)
    (heap : Heap) (inputBase outputBase M N : Nat) (hN : 0 < N) (address : Nat)
    (outside : address < outputBase ∨ outputBase + 4 * (M * N) ≤ address) :
    run ops compiler hB heap inputBase outputBase M N address = heap address := by
  unfold run
  rw [grid_range M N B hN hB]
  exact NativeStores.range_frame _ _ _ _ _ outside

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.NativeProgram
