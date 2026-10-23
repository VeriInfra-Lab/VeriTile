import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Allocation
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPReferenceGrid

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000
def flatIndex (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  ((c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val))*c.q.heads+(c.launch idx).head)*c.q.dims+i.2.1.val

theorem flat_index (c : Call) (r : Regions) (extent : RegionBounds)
    (g : Checked.geometryChecks c r extent) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) (on : FPReference.laneActive c idx i) :
    FPReference.laneOffset c idx i=flatIndex c idx i ∧ flatIndex c idx i<c.q.rows*c.q.heads*c.q.dims := by
  rcases g with ⟨_,_,k,v,o,contiguous,capacity,_⟩
  have bounds := (capacity ⟨(c.launch idx).batch,launch_batch_lt c idx⟩).1
  change c.start (c.launch idx).batch+c.length (c.launch idx).batch≤c.q.rows at bounds
  have row : c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val)<c.q.rows := by
    change _ < c.length _ at on
    omega
  have bounded := Checked.bounded c.q contiguous (c.q.rows*c.q.heads*c.q.dims) (le_refl _) _ _ _
    row (launch_head_lt c idx) i.2.1.isLt
  have eq := Checked.offset_eq c.q contiguous
    (c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val)) (c.launch idx).head i.2.1.val
  constructor
  · simpa only [FPReference.laneOffset,o,flatIndex] using eq
  · simpa only [eq,flatIndex] using bounded


end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
