import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealForwardFlat
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTraceBlocks

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open FPTrace TraceReadback
noncomputable section
set_option maxHeartbeats 6000000

abbrev valueCell (x : ℝ) : Cell (WithBot ℝ) := .mk .real (x : WithBot ℝ)

theorem online_inner_tile (s : State (WithBot ℝ)) (X Y : RegionName) (q : Nat → WithBot ℝ)
    (input : Input X q s) (separate : X≠Y) (masked : Bool) (row N B cur : Nat) (hB : 0<B)
    (valid : ∀ i : TileIndex [B], FPForwardInner.active masked N (FPForwardInner.previous N B-cur) i=Bool.true →
      FPForwardInner.previous N B-cur+i.1.val<N) :
    Sound (Input X q) (cells q N 1)
      (forwardInnerTile Ideal.algebra X Y masked row N B
        (FPForwardInner.maxReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B)))
        (FPForwardInner.sumReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B))) cur) := by
  intro t ht
  dsimp only [Input] at ht
  refine ⟨input_preserved X Y q _ rfl separate t ht,?_⟩
  intro i hi
  have hn := valid i hi
  change Cell.mk (α := WithBot ℝ) .real (FPForwardInner.storedValue Ideal.algebra t X masked row N B cur
    (FPForwardInner.maxReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B)))
    (FPForwardInner.sumReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B))) i) = _
  rw [online_inner_value s t X (fun o => (ht o).trans (input o).symm) masked row N B cur hB i hn]
  simpa only [cells,forwardInnerTile,FPForwardInner.offset,valueCell] using
    congrArg valueCell (inner_ratio_flat s X q input row N (FPForwardInner.previous N B-cur+i.1.val) hn)

theorem online_noninner_tile (s : State (WithBot ℝ)) (X Y : RegionName) (q : Nat → WithBot ℝ)
    (input : Input X q s) (separate : X≠Y) (row col N K B C cur : Nat) (hB : 0<B) :
    Sound (Input X q) (cells q N K)
      (forwardNoninnerTile Ideal.algebra X Y row col N K B C
        (FPForwardNoninner.maxReduced Ideal.algebra B C (noninnerModel s X row col N K B C (RealAddress.rounds N B)))
        (FPForwardNoninner.sumReduced Ideal.algebra B C (noninnerModel s X row col N K B C (RealAddress.rounds N B))) cur) := by
  intro t ht
  dsimp only [Input] at ht
  refine ⟨input_preserved X Y q _ rfl separate t ht,?_⟩
  intro i hi
  have active : FPForwardNoninner.previous N B-cur+i.1.val<N ∧ col*C+i.2.1.val<K := by
    simpa only [forwardNoninnerTile,FPForwardNoninner.active,Bool.and_eq_true,decide_eq_true_eq] using hi
  change Cell.mk (α := WithBot ℝ) .real (FPForwardNoninner.storedValue Ideal.algebra t X row col N K B C cur
    (FPForwardNoninner.maxReduced Ideal.algebra B C (noninnerModel s X row col N K B C (RealAddress.rounds N B)))
    (FPForwardNoninner.sumReduced Ideal.algebra B C (noninnerModel s X row col N K B C (RealAddress.rounds N B))) i) = _
  rw [online_noninner_value s t X (fun o => (ht o).trans (input o).symm) row col N K B C cur hB i active.1 active.2]
  simpa only [cells,forwardNoninnerTile,FPForwardNoninner.offset,Target.address,valueCell] using
    congrArg valueCell (noninner_ratio_flat s X q input row N K (col*C+i.2.1.val)
      (FPForwardNoninner.previous N B-cur+i.1.val) active.1 active.2)

theorem one_inner_tile (X Y : RegionName) (q : Nat → WithBot ℝ) (separate : X≠Y)
    (row N B : Nat) (hB : 0<B) (covered : N≤B)
    (defined : Target.Defined (fun n : Fin N => q (row*N+n.val))) :
    Sound (Input X q) (cells q N 1)
      ⟨[B],Y,FPReference.innerOffset row N,FPReference.innerActive N,
        fun s => FPReference.forwardInnerValue Ideal.algebra s X row N B⟩ := by
  intro t ht
  dsimp only [Input] at ht
  refine ⟨input_preserved X Y q _ rfl separate t ht,?_⟩
  intro i hi
  have hn : i.1.val<N := by simpa only [FPReference.innerActive,decide_eq_true_eq] using hi
  have hd : Target.Defined (fun n : Fin N => innerInput t X row N n.val) := by
    simpa only [innerInput,ht] using defined
  change Cell.mk (α := WithBot ℝ) .real (FPReference.forwardInnerValue Ideal.algebra t X row N B i) = _
  rw [one_inner_value t X row N B hB covered hd i hn]
  simpa only [cells,FPReference.innerOffset,valueCell] using
    congrArg valueCell (inner_ratio_flat t X q ht row N i.1.val hn)

theorem one_noninner_tile (X Y : RegionName) (q : Nat → WithBot ℝ) (separate : X≠Y)
    (row col N K B C : Nat) (hB : 0<B) (covered : N≤B)
    (defined : ∀ k, k<K → Target.Defined (Target.column q N K row k)) :
    Sound (Input X q) (cells q N K)
      ⟨[B,C],Y,FPReference.noninnerOffset row col N K,FPReference.noninnerActive col N K,
        fun s => FPReference.forwardNoninnerValue Ideal.algebra s X row col N K B C⟩ := by
  intro t ht
  dsimp only [Input] at ht
  refine ⟨input_preserved X Y q _ rfl separate t ht,?_⟩
  intro i hi
  have active : i.1.val<N ∧ col*C+i.2.1.val<K := by
    simpa only [FPReference.noninnerActive,Bool.and_eq_true,decide_eq_true_eq] using hi
  have hd : Target.Defined (fun n : Fin N => noninnerInput t X row N K (col*C+i.2.1.val) n.val) := by
    simpa only [Target.column,Target.address,noninnerInput,if_pos active.2,ht] using defined _ active.2
  change Cell.mk (α := WithBot ℝ) .real (FPReference.forwardNoninnerValue Ideal.algebra t X row col N K B C i) = _
  rw [one_noninner_value t X row col N K B C hB covered i active.1 active.2 hd]
  simpa only [cells,FPReference.noninnerOffset,Target.address,valueCell] using
    congrArg valueCell (noninner_ratio_flat t X q ht row N K (col*C+i.2.1.val) i.1.val active.1 active.2)

#axiomsClean online_inner_tile
#axiomsClean online_noninner_tile
#axiomsClean one_inner_tile
#axiomsClean one_noninner_tile
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
