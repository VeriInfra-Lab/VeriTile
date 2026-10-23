import bench.repairs.tritonbench_g.triton_argmax.Proofs.DimMemory
import bench.repairs.tritonbench_g.triton_argmax.Proofs.LoopLink
import bench.repairs.tritonbench_g.triton_argmax.Proofs.RealDim

/-! Full dimension numerical bridge. Its recursive trace condition lists only
reached tree-node comparisons and the reached scalar strict-update comparison.
All loop progress, index propagation and memory readback are conclusions. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.DimBridge
open Source Reduction
noncomputable section

def rowLogical (c : DimConfig) (logical : Nat → ℝ) (row inner : Nat) : Nat → ℝ :=
  fun col => logical (dimInputOffset c row col inner)

def realStep (c : DimConfig) (logical : Nat → ℝ) (row inner : Nat)
    (current : Math.Value × Nat) (tile : Nat) : Math.Value × Nat :=
  Math.strictStep c.extent (tile*blockN) blockN (rowLogical c logical row inner)
    (c.tree row inner tile) current

/-- Recursive enumeration of scalar obligations at actual reached states;
neither equality of intermediate reductions nor final outputs is a field. -/
def LocalNumerics {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) (row inner : Nat) :
    List Nat → Word × Nat → Math.Value × Nat → Prop
  | [],_,_ => True
  | tile::rest,raw,real =>
      (c.tree row inner tile).LocalAgreement ops.gt32 ops.eq32 orderedGt orderedEq
        (c.loaded bits ops.negInf32 row inner tile)
        (Math.padded c.extent (tile*blockN) blockN (rowLogical c logical row inner)) ∧
      ops.gt32 (dimLocal ops c bits row inner tile).1 raw.1 =
        orderedGt (Math.localPair c.extent (tile*blockN) blockN
          (rowLogical c logical row inner) (c.tree row inner tile)).1 real.1 ∧
      LocalNumerics ops c bits logical row inner rest (dimStep ops c bits row inner raw tile)
        (realStep c logical row inner real tile)

theorem local_index {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) (row inner tile : Nat)
    (num : (c.tree row inner tile).LocalAgreement ops.gt32 ops.eq32 orderedGt orderedEq
      (c.loaded bits ops.negInf32 row inner tile)
      (Math.padded c.extent (tile*blockN) blockN (rowLogical c logical row inner))) :
    (dimLocal ops c bits row inner tile).2=
      (Math.localPair c.extent (tile*blockN) blockN
        (rowLogical c logical row inner) (c.tree row inner tile)).2 := by
  have selected := pick_agrees ops.gt32 ops.eq32 orderedGt orderedEq _ _ _ num
  simp only [dimLocal,Math.localPair,Math.blockSelected,selected]

theorem fold_index {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) (row inner : Nat) (work : List Nat)
    (raw : Word × Nat) (real : Math.Value × Nat) (same : raw.2=real.2)
    (num : LocalNumerics ops c bits logical row inner work raw real) :
    (work.foldl (dimStep ops c bits row inner) raw).2=
      (work.foldl (realStep c logical row inner) real).2 := by
  induction work generalizing raw real with
  | nil => exact same
  | cons tile rest ih =>
    obtain ⟨tree,update,tail⟩ := num
    have localEq := local_index ops c bits logical row inner tile tree
    apply ih _ _ _ tail
    dsimp only [dimStep,realStep,Math.strictStep]
    rw [update]
    by_cases yes : real.1 < (Math.localPair c.extent (tile*blockN) blockN
      (rowLogical c logical row inner) (c.tree row inner tile)).1
    · simp only [orderedGt,yes,decide_true,ite_true]
      exact localEq
    · simp only [orderedGt,yes,decide_false,Bool.false_eq_true,ite_false]
      exact same

theorem real_scan (c : DimConfig) (logical : Nat → ℝ) (row inner count tile : Nat)
    (ends : tile+count=(c.extent+127)/128) (current : Math.Value × Nat) :
    Math.dimensionLoop c.extent 128 (by omega) (rowLogical c logical row inner)
      (fun cur => c.tree row inner (cur/128)) (tile*128) current=
      LoopLink.scan (fun t a => realStep c logical row inner a t) count tile current := by
  induction count generalizing tile current with
  | zero =>
    have inactive : ¬tile*128 < c.extent := by rw [←LoopLink.ceil_tiles]; omega
    rw [Math.dimensionLoop,dif_neg inactive]
    rfl
  | succ count ih =>
    have active : tile*128 < c.extent := by rw [←LoopLink.ceil_tiles]; omega
    rw [Math.dimensionLoop,dif_pos active]
    have next : tile*128+128=(tile+1)*128 := by omega
    have div : tile*128/128=tile := by omega
    rw [next,div]
    exact ih (tile+1) (by omega) _

theorem dimension_index {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) (row inner : Nat)
    (num : LocalNumerics ops c bits logical row inner (dimTiles c) (ops.negInf32,0) (⊥,0)) :
    dimIndex ops c bits row inner=Math.dimensionIndex c.extent 128 (by omega)
      (rowLogical c logical row inner) (fun cur => c.tree row inner (cur/128)) := by
  have paired := fold_index ops c bits logical row inner (dimTiles c) (ops.negInf32,0) (⊥,0) rfl num
  have scheduled := real_scan c logical row inner ((c.extent+127)/128) 0 (by omega) (⊥,0)
  simp only [Nat.zero_mul,LoopLink.scan_fold] at scheduled
  unfold Math.dimensionIndex
  rw [scheduled]
  simpa only [dimIndex,dimTiles,blockN,List.range_eq_range',
    show c.extent+128-1=c.extent+127 by omega] using paired

theorem reference_index {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) (row inner : Nat) (nonempty : 0 < c.extent)
    (complete : ∀ tile,(c.tree row inner tile).Complete)
    (num : LocalNumerics ops c bits logical row inner (dimTiles c) (ops.negInf32,0) (⊥,0)) :
    dimIndex ops c bits row inner=(Math.argmax c.extent nonempty
      (fun i : Fin c.extent => rowLogical c logical row inner i.val)).val := by
  rw [dimension_index ops c bits logical row inner num]
  obtain ⟨bound,correct⟩ := Math.dimension_mathematical c.extent 128 nonempty (by omega)
    (rowLogical c logical row inner) (fun cur => c.tree row inner (cur/128)) (fun cur _ => complete _)
  have unique := least_maximum_unique (fun i : Fin c.extent => rowLogical c logical row inner i.val)
    ⟨_,bound⟩ (Math.argmax c.extent nonempty (fun i : Fin c.extent => rowLogical c logical row inner i.val))
    correct (Math.argmax_spec _ _ _)
  exact congrArg Fin.val unique

theorem returned_bytes {M : Type} {d : MidFormat} (mem : Memory.Primitives M)
    (ops : Scalars d) (c : DimConfig) (p : DimMemory.Buffers) (layout : DimMemory.Layout c p)
    (s : M) (logical : Nat → ℝ) (row inner : Nat) (hr : row < c.outer) (hi : inner < c.inner)
    (nonempty : 0 < c.extent) (complete : ∀ tile,(c.tree row inner tile).Complete)
    (num : LocalNumerics ops c (DimMemory.input mem p s) logical row inner (dimTiles c) (ops.negInf32,0) (⊥,0)) :
    mem.read 8 (DimMemory.run mem ops c p s) (Memory.address p.output 8 (dimOutputOffset c row inner))=
      BitVec.ofNat 64 (Math.argmax c.extent nonempty
        (fun i : Fin c.extent => rowLogical c logical row inner i.val)).val := by
  rw [DimMemory.dimension_readback mem ops c p layout s row inner hr hi,
    reference_index ops c (DimMemory.input mem p s) logical row inner nonempty complete num]

#axiomsClean fold_index
#axiomsClean dimension_index
#axiomsClean reference_index
#axiomsClean returned_bytes
end
end VeriTile.Bench.Repairs.TritonArgmax.DimBridge
