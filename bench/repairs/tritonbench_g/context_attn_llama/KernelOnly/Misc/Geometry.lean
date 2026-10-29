import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Allocation

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 6000000

def flatIndex (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  ((c.start (Formula.batch c.layout (c.launchAt idx))+Formula.queryRow c.layout (c.launchAt idx) i.1.val)*c.layout.heads+
    Formula.head c.layout (c.launchAt idx))*c.layout.D+i.2.1.val

theorem flat_index (c : Call) (d : Checked.Dimensions) (extent : RegionBounds)
    (g : Checked.geometryChecks c d extent) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D])
    (on : c.active idx i) : c.offset idx i=flatIndex c idx i ∧ flatIndex c idx i < d.queryRows*c.layout.heads*c.layout.D := by
  have cap := (g.2.2.2.2.2.2.1 ⟨Formula.batch c.layout (c.launchAt idx),Host.batch_bound c idx⟩).1
  have row : c.start (Formula.batch c.layout (c.launchAt idx))+Formula.queryRow c.layout (c.launchAt idx) i.1.val < d.queryRows := by
    change c.start (Formula.batch c.layout (c.launchAt idx))+c.length (Formula.batch c.layout (c.launchAt idx))≤d.queryRows at cap
    change Formula.queryRow _ _ _ < c.length _ at on
    omega
  have bound := Checked.bound3 d.queryRows c.layout.heads c.layout.D _ _ _ _ (le_refl _)
    row (Host.head_bound c idx) i.2.1.isLt
  rcases g.1 with ⟨_,_,_,orow,ohead,odim,_⟩
  have eq : c.offset idx i=flatIndex c idx i := by
    simp only [Call.offset,Call.output,orow,ohead,odim,Nat.mul_one,flatIndex]
    ring
  exact ⟨eq,by simpa only [flatIndex,Nat.add_mul,Nat.mul_assoc] using bound⟩

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
