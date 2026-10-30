import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOGeometry
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteOperands
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteComplete

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOStoreGeometry
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 2000000

theorem active_iff {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (i : TileIndex [c.BM,c.BN]) :
    Quote.active s c (preludeExecution M q s c hasBias initial).val i ↔
      Geometry.Active (geometry s c) (initial.pids 0) i := by
  unfold Quote.active
  rw [prelude_rm M q s c hasBias initial valid,prelude_rn M q s c hasBias initial valid]
  simp only [Int.ofNat_lt]
  rfl

theorem offset_eq {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (i : TileIndex [c.BM,c.BN]) :
    Quote.offset s c (preludeExecution M q s c hasBias initial).val i =
      row s c (initial.pids 0) i.1*s.N+col s c (initial.pids 0) i.2.1 := by
  unfold Quote.offset
  rw [prelude_rm M q s c hasBias initial valid,prelude_rn M q s c hasBias initial valid]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem offset_injective {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (i j : TileIndex [c.BM,c.BN])
    (hi : Quote.active s c (preludeExecution M q s c hasBias initial).val i)
    (hj : Quote.active s c (preludeExecution M q s c hasBias initial).val j)
    (same : Quote.offset s c (preludeExecution M q s c hasBias initial).val i=
      Quote.offset s c (preludeExecution M q s c hasBias initial).val j) : i=j := by
  have ai := (active_iff M q s c hasBias initial valid i).mp hi
  have aj := (active_iff M q s c hasBias initial valid j).mp hj
  apply Geometry.offset_injective (geometry s c) (Grouped.valid s c) (initial.pids 0) i j ai aj
  rw [offset_eq M q s c hasBias initial valid,offset_eq M q s c hasBias initial valid] at same
  simpa only [Geometry.offset,geometry,Nat.mul_one] using same

theorem active_cell {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (i : TileIndex [c.BM,c.BN])
    (enabled : Quote.active s c (preludeExecution M q s c hasBias initial).val i) :
    ∃ cell : MatrixCell s,
      (Quote.offset s c (preludeExecution M q s c hasBias initial).val i=cell.1.val*s.N+cell.2.val) ∧
      Geometry.Owns (geometry s c) (initial.pids 0) cell := by
  have a := (active_iff M q s c hasBias initial valid i).mp enabled
  refine ⟨Geometry.index (geometry s c) (initial.pids 0) i a,?_,i,rfl,rfl⟩
  exact offset_eq M q s c hasBias initial valid i

theorem offset_miss {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (cell : MatrixCell s)
    (other : initial.pids 0≠(IOPrograms.owner s c cell).val) (i : TileIndex [c.BM,c.BN])
    (enabled : Quote.active s c (preludeExecution M q s c hasBias initial).val i) :
    cell.1.val*s.N+cell.2.val≠Quote.offset s c (preludeExecution M q s c hasBias initial).val i := by
  intro same
  obtain ⟨found,address,owns⟩:=active_cell M q s c hasBias initial valid i enabled
  have equal : cell=found := by
    apply outputAddress_injective s (Buffers.mk 0 0 0 0 0 0 0 0 0 0 0 0)
    apply Prod.ext
    · rfl
    · simpa [outputAddress] using same.trans address
  subst found
  have owner:=IOGeometry.owner_unique s c cell ⟨initial.pids 0,valid⟩ owns
  exact other (congrArg Fin.val owner)

#axiomsClean active_iff
#axiomsClean offset_eq
#axiomsClean offset_injective
#axiomsClean active_cell
#axiomsClean offset_miss
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOStoreGeometry
