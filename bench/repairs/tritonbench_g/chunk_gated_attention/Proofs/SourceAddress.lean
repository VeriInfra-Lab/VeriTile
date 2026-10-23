import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.SourcePointer
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CumulativeCore

/-! Source-specific finite-pointer certificate over the very buffers returned
by successful materialization. All eleven local pointer occurrences have zero
block offsets. Each certificate relates the actual wide base/lane expression,
its dtype-scaled native address, and one typed cell of that same allocation. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.SourceAddress
open VeriTile.Triton HostFP Address
noncomputable section
set_option maxHeartbeats 3000000

theorem inner_allocated {α : Type} 
    (x : HostRequest.Inner α) (r : HostRequest.Regions) (hx : x.Valid)
    (p : HostRequest.Allocated x r) (i : InnerCoord (x.config r))
    (site : InnerSite) (he : enabled (x.config r) site) :
    innerOffset (x.config r) i site<(p.buffers (region r site)).extent := by
  have hbt := hx.bt
  have hbk := hx.bk
  have hbv := hx.bv
  have hb := inner_bounds (x.config r) i (by change 0<x.BT;omega) (by change 0<x.BK;omega) (by change 0<x.BV;omega)
  have hk := Address.flatten3_bounded i.head.isLt hb.2.2.1 hb.1
  have hv := Address.flatten3_bounded i.head.isLt hb.2.2.1 hb.2.1
  have hke := Address.flatten3_bounded i.head.isLt hb.2.2.2 hb.1
  have hve := Address.flatten3_bounded i.head.isLt hb.2.2.2 hb.2.1
  have hs := Address.flatten3_bounded i.head.isLt hb.1 hb.2.1
  have hh := Address.history_bounded i.head.isLt i.chunk.isLt hb.1 hb.2.1
  cases site with
  | key => exact lt_of_lt_of_le hk p.keyExtent
  | value => exact lt_of_lt_of_le hv p.valueExtent
  | history => exact lt_of_lt_of_le hh p.historyExtent
  | seed => exact lt_of_lt_of_le hs (p.seedExtent he)
  | final => exact lt_of_lt_of_le hs (p.finalExtent he)
  | gateKey =>
    have hg : x.gateK=Bool.true := he
    exact lt_of_lt_of_le hk (by simpa [hg] using p.gateExtent)
  | gateKeyEnd =>
    have hg : x.gateK=Bool.true := he
    exact lt_of_lt_of_le hke (by simpa [hg] using p.gateExtent)
  | gateValue =>
    have hg : x.gateK=Bool.false := he
    exact lt_of_lt_of_le hv (by simpa [hg] using p.gateExtent)
  | gateValueEnd =>
    have hg : x.gateK=Bool.false := he
    exact lt_of_lt_of_le hve (by simpa [hg] using p.gateExtent)

def innerAccess {α : Type} 
    (x : HostRequest.Inner α) (r : HostRequest.Regions) (hx : x.Valid)
    (p : HostRequest.Allocated x r) (i : InnerCoord (x.config r))
    (site : InnerSite) (he : enabled (x.config r) site) : Access (p.buffers (region r site)) :=
  ⟨innerOffset (x.config r) i site,innerWord (x.config r) i site,
    inner_word _ _ _ (by change 0<x.BT;have := hx.bt;omega),inner_allocated x r hx p i site he⟩

#print axioms inner_allocated
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.SourceAddress
