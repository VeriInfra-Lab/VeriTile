import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardValue

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open FPTrace TraceReadback
noncomputable section
set_option maxHeartbeats 6000000

def cells (p g : Nat → ℝ) (N K : Nat) (_ : RegionName) (o : Nat) : Cell (WithBot ℝ) :=
  .mk .real (Target.flatBackward p g N K o : WithBot ℝ)

theorem read_preserved (R Y : RegionName) (v : Nat → ℝ) (count : Nat)
    (tile : Tile (WithBot ℝ)) (region : tile.region=Y) (separate : R≠Y)
    (s : State (WithBot ℝ)) (read : Read R v count s) : Read R v count (tile.execute s) := by
  intro o ho
  change ((tile.execute s).mem R o).read .real=(v o : WithBot ℝ)
  rw [Tile.frame]
  · exact read o ho
  · intro hit
    obtain ⟨i,hi,he⟩ := (Tile.mem_events tile (R,o)).mp hit
    exact separate ((congrArg Prod.fst he).trans region)

theorem input_preserved (P G Y : RegionName) (p g : Nat → ℝ) (count : Nat)
    (tile : Tile (WithBot ℝ)) (region : tile.region=Y) (hPY : P≠Y) (hGY : G≠Y)
    (s : State (WithBot ℝ)) (input : Input P G p g count s) : Input P G p g count (tile.execute s) :=
  ⟨read_preserved P Y p count tile region hPY s input.1,
    read_preserved G Y g count tile region hGY s input.2⟩

theorem online_inner_tile (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (row rows N B C cur : Nat) (hB : 0<B) :
    Sound (Input P G p g (rows*N)) (cells p g N 1)
      (backwardInnerTile Ideal.algebra P G Y row rows N B C
        (FPBackwardInner.reduced Ideal.algebra B C (innerModel p g row rows N B C (RealAddress.rounds N B))) cur) := by
  intro t ht
  refine ⟨input_preserved P G Y p g _ _ rfl hPY hGY t ht,?_⟩
  intro i hi
  have active : row*C+i.1.val<rows ∧ cur+i.2.1.val<N := by
    simpa only [backwardInnerTile,FPBackwardInner.active,Bool.and_eq_true,decide_eq_true_eq] using hi
  change Cell.mk (α := WithBot ℝ) .real (FPBackwardInner.storedValue Ideal.algebra t P G row rows N B C cur
    (FPBackwardInner.reduced Ideal.algebra B C (innerModel p g row rows N B C (RealAddress.rounds N B))) i)=_
  rw [online_inner_value t P G p g row rows N B C cur hB ht i active.1 active.2]
  rfl

theorem online_noninner_tile (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (row col rows N K B C cur : Nat) (hB : 0<B) (hr : row<rows) :
    Sound (Input P G p g (rows*N*K)) (cells p g N K)
      (backwardNoninnerTile Ideal.algebra P G Y row col rows N K B C
        (FPBackwardNoninner.reduced Ideal.algebra B C (noninnerModel p g row col N K B C (RealAddress.rounds N B))) cur) := by
  intro t ht
  refine ⟨input_preserved P G Y p g _ _ rfl hPY hGY t ht,?_⟩
  intro i hi
  have active : cur+i.1.val<N ∧ col*C+i.2.1.val<K := by
    simpa only [backwardNoninnerTile,FPBackwardNoninner.active,Bool.and_eq_true,decide_eq_true_eq] using hi
  change Cell.mk (α := WithBot ℝ) .real (FPBackwardNoninner.storedValue Ideal.algebra t P G row col rows N K B C cur
    (FPBackwardNoninner.reduced Ideal.algebra B C (noninnerModel p g row col N K B C (RealAddress.rounds N B))) i)=_
  rw [online_noninner_value t P G p g row col rows N K B C cur hB hr ht i active.1 active.2]
  rfl

theorem one_inner_tile (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (row rows N B C : Nat) (hB : 0<B) (covered : N≤B) :
    Sound (Input P G p g (rows*N)) (cells p g N 1)
      ⟨[C,B],Y,FPReference.backwardInnerOffset row N,FPReference.backwardInnerActive row rows N,
        fun s => FPReference.backwardInnerValue Ideal.algebra s P G row rows N B C⟩ := by
  intro t ht
  refine ⟨input_preserved P G Y p g _ _ rfl hPY hGY t ht,?_⟩
  intro i hi
  have active : row*C+i.1.val<rows ∧ i.2.1.val<N := by
    simpa only [FPReference.backwardInnerActive,Bool.and_eq_true,decide_eq_true_eq] using hi
  change Cell.mk (α := WithBot ℝ) .real (FPReference.backwardInnerValue Ideal.algebra t P G row rows N B C i)=_
  rw [one_inner_value t P G p g row rows N B C hB covered ht i active.1 active.2]
  rfl

theorem one_noninner_tile (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (row col rows N K B C : Nat) (hB : 0<B) (covered : N≤B) (hr : row<rows) :
    Sound (Input P G p g (rows*N*K)) (cells p g N K)
      ⟨[B,C],Y,FPReference.noninnerOffset row col N K,FPReference.noninnerActive col N K,
        fun s => FPReference.backwardNoninnerValue Ideal.algebra s P G row col N K B C⟩ := by
  intro t ht
  refine ⟨input_preserved P G Y p g _ _ rfl hPY hGY t ht,?_⟩
  intro i hi
  have active : i.1.val<N ∧ col*C+i.2.1.val<K := by
    simpa only [FPReference.noninnerActive,Bool.and_eq_true,decide_eq_true_eq] using hi
  change Cell.mk (α := WithBot ℝ) .real (FPReference.backwardNoninnerValue Ideal.algebra t P G row col N K B C i)=_
  rw [one_noninner_value t P G p g row col rows N K B C hB covered hr ht i active.1 active.2]
  rfl

#axiomsClean online_inner_tile
#axiomsClean online_noninner_tile
#axiomsClean one_inner_tile
#axiomsClean one_noninner_tile
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
