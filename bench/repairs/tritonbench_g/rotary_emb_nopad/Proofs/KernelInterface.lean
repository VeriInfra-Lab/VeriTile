import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedNative

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.KernelInterface
structure Geometry where
  (tokens qHeads kHeads half cosWidth sinWidth blockSize tableColumns : Nat)

def Geometry.request (g : Geometry) : Request where
  tokens:=g.tokens; qHeads:=g.qHeads; kHeads:=g.kHeads; half:=g.half
  cosRows:=0; cosWidth:=g.cosWidth; sinRows:=0; sinWidth:=g.sinWidth
  cached:=Bool.false; blocks:=0; blockSize:=g.blockSize; tableColumns:=g.tableColumns
  device:=0; qBase:=0; kBase:=0; cosBase:=0; sinBase:=0; cacheBase:=0; tableBase:=0; lengthBase:=0

def geometry (r : Request) : Geometry :=
  ⟨r.tokens,r.qHeads,r.kHeads,r.half,r.cosWidth,r.sinWidth,r.blockSize,r.tableColumns⟩

theorem uncached_layout (r : Request) : Typed.uLayout (geometry r).request=Typed.uLayout r := rfl
theorem cached_layout (r : Request) : Typed.cLayout (geometry r).request=Typed.cLayout r := rfl
end VeriTile.Bench.Repairs.RotaryEmbNopad.KernelInterface
