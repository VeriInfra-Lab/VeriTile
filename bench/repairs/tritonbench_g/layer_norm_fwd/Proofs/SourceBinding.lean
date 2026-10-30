import bench.repairs.tritonbench_g.layer_norm_fwd.Kernels
import bench.tritonbench_g.layer_norm_fwd.Proofs.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd

abbrev repaired : sourceType := original
abbrev rowBits := Source.rowBits
abbrev scalarBits := Source.scalarBits
abbrev seedStorageBits := Source.seedStorageBits
abbrev randomRounds := Source.randomRounds

def sourceFlags (f : Flags) : Source.Flags :=
  ⟨f.rms,f.residual,f.residualOut,f.bias,f.dropout,f.dropoutMask,f.rowscale,f.x1,f.w1,f.b1⟩
def sourceRegions (r : Regions) : Source.Regions :=
  ⟨r.x,r.y,r.weight,r.bias,r.residual,r.x1,r.weight1,r.bias1,r.y1,
    r.residualOut,r.rowscale,r.seeds,r.dropoutMask,r.mean,r.rstd⟩
def sourceStrides (s : Strides) : Source.Strides :=
  ⟨s.x,s.y,s.residual,s.residualOut,s.x1,s.y1⟩

theorem source_exact (f : Flags) (r : Regions) (s : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) :
    Source.kernel (sourceFlags f) (sourceRegions r) (sourceStrides s) M N eps p block hb =
      repaired f r s M N eps p block hb := rfl

#axiomsClean source_exact
end VeriTile.Bench.Repairs.LayerNormFwd
