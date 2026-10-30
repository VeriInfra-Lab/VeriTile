import bench.tritonbench_g.attention_llama.Proofs.Output
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural
noncomputable section

/-- The four actual masked source footprints, before logical IO capacities
are connected. No bound is imposed on a masked-out pointer. -/
structure AccessBounds (c : Config) (s : State (Term α)) (bounds : RegionName→Nat) : Prop where
  query : ∀ i : TileIndex [64,c.D], rows s i.1<c.H →
    base c s c.sqz c.sqh+rows s i.1*c.sqm+i.2.1.val*c.sqk<bounds c.Q
  key : ∀ n (i : TileIndex [64,c.D]), n*64+i.1.val<c.N →
    (pointers c s c.K c.skz c.skh c.skn c.skk n i).2<bounds c.K
  value : ∀ n (i : TileIndex [64,c.D]), n*64+i.1.val<c.N →
    (pointers c s c.V c.svz c.svh c.svk c.svn n i).2<bounds c.V
  output : ∀ i : TileIndex [64,c.D], rows s i.1<c.H →
    address c s i<bounds c.Out

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
