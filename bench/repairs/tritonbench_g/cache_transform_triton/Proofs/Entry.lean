import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Tensor

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 100000

/-- Ordinary successful setup facts, not an output or kernel-success premise.
Metadata stores the row list computed and proved in Math. DirectView bytes are
read-only, typed uint8 and coordinate-preserving by the library contract. -/
structure EntryBinding {α β : Type} [Inhabited α] (p : Parameters) (s : State α)
    (mode : Mode) (lengths : List Nat) (v : LogicalView β) (width : Nat)
    (observe : (Fin v.dtype.bytes → Nat) → β) (direct : DirectView v width observe) : Prop where
  count : p.count = (preparedRows mode lengths).length * (width * v.dtype.bytes)
  rowBytes : p.rowBytes = width * v.dtype.bytes
  freshCache : p.cache ≠ p.output
  freshRows : p.rows ≠ p.output
  metadata : 0 < p.count → ∀ q, q < (preparedRows mode lengths).length →
    s.mem p.rows q = .mk .int (tableRows mode lengths q : Int)
  payload : ∀ o, o < v.rows * p.rowBytes → s.mem p.cache o = .mk .nat (direct.bytes o)
  dimension : v.rows ≤ i64Max
  /-- These extent facts come from successful ordinary Torch allocations.
  The current wrapper checks the byte extents explicitly. -/
  outputExtent : p.count ≤ i64Max
  cacheExtent : v.rows * p.rowBytes ≤ i64Max
  metadataExtent : 0 < p.count → (preparedRows mode lengths).length * 8 ≤ i64Max

theorem entry_gather_input {α β : Type} [Inhabited α] (p : Parameters) (s : State α)
    (mode : Mode) (lengths : List Nat) (v : LogicalView β) (width : Nat)
    (observe : (Fin v.dtype.bytes → Nat) → β) (direct : DirectView v width observe)
    (binding : EntryBinding p s mode lengths v width observe direct)
    (valid : ValidLengths mode lengths v.rows) (hpos : 0 < p.count) :
    GatherInput p s (preparedRows mode lengths).length v.rows
      (tableRows mode lengths) direct.bytes := by
  refine ⟨?_, ?_, binding.freshCache, binding.freshRows, binding.metadata hpos,
    tableRows_bound mode lengths v.rows valid, binding.dimension, binding.payload⟩
  · rw [binding.count, binding.rowBytes]
  · rw [binding.count] at hpos
    rw [binding.rowBytes]
    exact Nat.pos_of_mul_pos_left hpos

/-- Whole returned tensor pair: actual typed kernel executions, independent
sequence mathematics, all rows/columns, and a complete memory frame. -/
theorem whole_call {α β : Type} [Inhabited α] (M : Algebra α)
    (cos sin : Parameters) (s : State α) (mode : Mode) (lengths : List Nat)
    (cv sv : LogicalView β) (width : Nat)
    (cosObserve : (Fin cv.dtype.bytes → Nat) → β)
    (sinObserve : (Fin sv.dtype.bytes → Nat) → β)
    (cs : DirectView cv width cosObserve) (ss : DirectView sv width sinObserve)
    (cb : EntryBinding cos s mode lengths cv width cosObserve cs)
    (sb : EntryBinding sin s mode lengths sv width sinObserve ss)
    (validCos : ValidLengths mode lengths cv.rows)
    (validSin : ValidLengths mode lengths sv.rows)
    (freshOutput : cos.output ≠ sin.output)
    (freshCos : sin.cache ≠ cos.output) (freshRows : sin.rows ≠ cos.output) :
    let t := pairGather M cos sin s
    (∀ k, k < blocks cos.count →
      FP.Structural.exec M (source (tileParameters cos k))
        (tileState (runTiles M cos s k) k) = some (runTiles M cos s (k + 1))) ∧
    (∀ k, k < blocks sin.count →
      FP.Structural.exec M (source (tileParameters sin k))
        (tileState (runTiles M sin (gather M cos s) k) k) =
          some (runTiles M sin (gather M cos s) (k + 1))) ∧
    observedTensor t cos.output (preparedRows mode lengths).length width cv.dtype cosObserve =
      mathematical mode lengths (fun r => fun c : Fin width => cv.logical r c.val) ∧
    observedTensor t sin.output (preparedRows mode lengths).length width sv.dtype sinObserve =
      mathematical mode lengths (fun r => fun c : Fin width => sv.logical r c.val) ∧
    (∀ r o, (r ≠ cos.output ∨ o ≥ cos.count) → (r ≠ sin.output ∨ o ≥ sin.count) →
      t.mem r o = s.mem r o) := by
  have h := pair_correct M cos sin s (preparedRows mode lengths).length cv.rows sv.rows
    (tableRows mode lengths) cs.bytes ss.bytes
    (entry_gather_input cos s mode lengths cv width cosObserve cs cb validCos)
    (entry_gather_input sin s mode lengths sv width sinObserve ss sb validSin)
    freshOutput freshCos freshRows
  refine ⟨?_, ?_, ?_, ?_, h.2.2⟩
  · intro k _
    exact tile_success M cos k _
  · intro k _
    exact tile_success M sin k _
  · exact (tensor_from_bytes cos _ mode lengths cs.bytes width cv.dtype cosObserve
      cb.count cb.rowBytes h.1).trans (direct_view_math cv width cosObserve cs mode lengths validCos)
  · exact (tensor_from_bytes sin _ mode lengths ss.bytes width sv.dtype sinObserve
      sb.count sb.rowBytes h.2.1).trans (direct_view_math sv width sinObserve ss mode lengths validSin)

#axiomsClean entry_gather_input
#axiomsClean whole_call
end
end VeriTile.Bench.Repairs.CacheTransform
