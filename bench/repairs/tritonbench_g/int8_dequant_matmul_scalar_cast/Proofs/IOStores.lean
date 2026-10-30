import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOStoreFold
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOStoreGeometry
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOValue

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOStores
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem stores_at (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c)) (divisor : F)
    (i : TileIndex [c.BM,c.BN])
    (enabled : Quote.active s c (preludeExecution (floating q) q s c hasBias initial).val i) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    Quote.stores q s c hasBias base divisor "C" (Quote.offset s c base i)=
      Cell.mk .fp16 (.inr (if c.S=1 then Quote.value q s c hasBias base divisor i else
        q.ops.hadd (Quote.asH q.ops ((initial.mem "C" (Quote.offset s c base i)).read .fp16))
          (Quote.value q s c hasBias base divisor i))) := by
  let base:=(preludeExecution (floating q) q s c hasBias initial).val
  let combine : Cell (Sum F H)→TileIndex [c.BM,c.BN]→Cell (Sum F H) := fun old j=>
    .mk .fp16 (.inr (if c.S=1 then Quote.value q s c hasBias base divisor j else
      q.ops.hadd (Quote.asH q.ops (old.read .fp16)) (Quote.value q s c hasBias base divisor j)))
  have read:=IOStoreFold.read (Quote.active s c base) (Quote.offset s c base) combine "C"
    (TileShape.allIndices [c.BM,c.BN]) base.mem (TileShape.allIndices_nodup _) i
    (TileShape.mem_allIndices _ i) enabled
    (fun j _ hj same=>IOStoreGeometry.offset_injective (floating q) q s c hasBias initial valid i j enabled hj same)
  simpa only [IOStoreFold.updates,combine,Quote.stores,base,prelude_memory] using read

theorem stores_miss (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State (Sum F H)) (divisor : F)
    (region : RegionName) (offset : Nat)
    (miss : region≠("C":RegionName) ∨ ∀ i : TileIndex [c.BM,c.BN],
      Quote.active s c (preludeExecution (floating q) q s c hasBias initial).val i →
      offset≠Quote.offset s c (preludeExecution (floating q) q s c hasBias initial).val i) :
    Quote.stores q s c hasBias (preludeExecution (floating q) q s c hasBias initial).val divisor
      region offset=initial.mem region offset := by
  let base:=(preludeExecution (floating q) q s c hasBias initial).val
  let combine : Cell (Sum F H)→TileIndex [c.BM,c.BN]→Cell (Sum F H) := fun old j=>
    .mk .fp16 (.inr (if c.S=1 then Quote.value q s c hasBias base divisor j else
      q.ops.hadd (Quote.asH q.ops (old.read .fp16)) (Quote.value q s c hasBias base divisor j)))
  have read:=IOStoreFold.frame (Quote.active s c base) (Quote.offset s c base) combine "C"
    (TileShape.allIndices [c.BM,c.BN]) base.mem region offset
    (miss.imp_right (fun h i _=>h i))
  simpa only [IOStoreFold.updates,combine,Quote.stores,base,prelude_memory] using read

theorem stores_frame (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c)) (divisor : F)
    (final : State (Sum F H))
    (memory : final.mem=Quote.stores q s c hasBias
      (preludeExecution (floating q) q s c hasBias initial).val divisor) :
    (IOLayout.memory F H s c hasBias).Frame vals initial final := by
  intro region offset outside
  rw [memory]
  apply stores_miss
  by_cases different : region≠("C":RegionName)
  · exact Or.inl different
  · right
    intro i enabled same
    obtain ⟨cell,address,_⟩:=IOStoreGeometry.active_cell (floating q) q s c hasBias initial valid i enabled
    have out:=outside.1 0 (cell.1,cell.2,PUnit.unit) trivial
    change region≠("C":RegionName) ∨
      offset≠TileShape.linearIndex [s.M,s.N] (cell.1,cell.2,PUnit.unit) at out
    simp only [TileShape.linearIndex,TileShape.product,Nat.mul_one,Nat.add_zero] at out
    exact out.elim (fun h=>h (not_ne_iff.mp different)) (fun h=>h (same.trans address))

#axiomsClean stores_at
#axiomsClean stores_miss
#axiomsClean stores_frame
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOStores
