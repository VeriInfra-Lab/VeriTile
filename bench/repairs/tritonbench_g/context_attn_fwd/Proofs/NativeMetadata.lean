import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeAddresses

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

/-- Unconditional contiguous INT32 metadata loads use the same exact batch
quotient as Q/K/V/Out. This covers all three prelude memory events. -/
def metadataEvent (c : Host.Call) (B : Bytes extent) (rg : RegionName) (idx : GridIndex c.grid) :=
  access Bool.true (B.base rg) 4 (batchBits c.layout (c.launchAt idx))

theorem metadata_event (c : Host.Call) (B : Bytes extent) (hc : Geometry.Checked c extent)
    (abi : ABI c) (rg : RegionName)
    (hr : rg∈[c.regions.start,c.regions.sequence,c.regions.prompt]) (idx : GridIndex c.grid) :
    metadataEvent c B rg idx=some (B.base rg+4*batchAt c idx) := by
  have hh := launch_abi c abi idx
  have he := (launch_exact c.layout (c.launchAt idx) (batchHead_abi c abi idx) hh.1 hh.2).1
  have hb := Host.batch_bound c idx
  have ho : batchAt c idx<extent rg := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl
    · exact lt_of_lt_of_le hb hc.start_span
    · exact lt_of_lt_of_le hb hc.sequence_span
    · exact lt_of_lt_of_le hb hc.prompt_span
  unfold metadataEvent
  rw [he,access_exact Bool.true _ 4 (batchAt c idx) (extent rg) (by decide) (B.fits rg) (fun _ => ho)]
  rfl
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
