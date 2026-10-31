import bench.tritonbench_g.attention_fwd_triton3.Sources.Selected
import bench.tritonbench_g.attention_fwd_triton3.Sources.Kernels

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Kernels
open VeriTile.Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

def source (c : Call) : ComputeKernel :=
  Quoted.kernel "Q" "K" "V" "M" "O" "L" c.scale
    c.q.batch c.q.head c.q.row c.q.dim
    c.k.batch c.k.head c.k.row c.k.dim
    c.v.batch c.v.head c.v.row c.v.dim
    c.o.batch c.o.head c.o.row c.o.dim
    c.Z c.H c.HK c.NQ c.rounded c.NK
    c.params.evenM c.params.evenN c.BM c.D c.BN c.finish c.init c.window c.complement c.offset c.size

def keyMaskCSE (c : Call) : ComputeKernel :=
  Quoted.keyMaskCSE "Q" "K" "V" "M" "O" "L" c.scale
    c.q.batch c.q.head c.q.row c.q.dim
    c.k.batch c.k.head c.k.row c.k.dim
    c.v.batch c.v.head c.v.row c.v.dim
    c.o.batch c.o.head c.o.row c.o.dim
    c.Z c.H c.HK c.NQ c.rounded c.NK
    c.params.evenM c.params.evenN c.BM c.D c.BN c.finish c.init c.window c.complement c.offset c.size

/-- The scalar conversion fix leaves the mathematical shared quotation exact.
This equality connects the candidate source to the preserved proof input. -/
theorem source_eq_inherited (c : Call) : source c =
    VeriTile.Bench.Repairs.AttentionFwdTriton3.Kernels.source c := rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Kernels
