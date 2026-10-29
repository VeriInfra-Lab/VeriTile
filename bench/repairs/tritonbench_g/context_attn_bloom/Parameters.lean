import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.HostParameters

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Raw
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom
noncomputable section
set_option autoImplicit false

/-- Raw three-axis extents and scalar element strides. -/
structure View where
  rows : Nat
  heads : Nat
  dims : Nat
  rowStride : Nat
  headStride : Nat
  dimStride : Nat

def View.toHost (v : View) : VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3 :=
  ⟨v.rows,v.heads,v.dims,v.rowStride,v.headStride,v.dimStride,Bool.false⟩

/-- Kernel launch geometry and the passed scale; no Tensor API state. -/
structure Geometry where
  q : View
  k : View
  v : View
  o : View
  requestRows : Nat
  requestColumns : Nat
  requestRowStride : Nat
  requestTokenStride : Nat
  batch : Nat
  maxInput : Nat
  block64 : Bool
  scale : ℝ

def Geometry.arguments (g : Geometry) : Host.Arguments :=
  ⟨g.q.toHost,g.k.toHost,g.v.toHost,g.o.toHost,g.requestRows,g.requestColumns,
    g.requestRowStride,g.requestTokenStride,g.batch,g.maxInput,g.block64,
    ⟨fun _ => 1,fun _ _ => g.scale,fun x y => x*y⟩⟩

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Raw
