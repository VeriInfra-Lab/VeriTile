import bench.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Geometry
def rows (s : Shape) : Region → Nat
  | .h0 | .ht | .dht | .dh0 => s.K
  | _ => s.T
def cols (s : Shape) (f : Flags) : Region → Nat
  | .q | .k | .dq | .dk | .retq | .retk => s.K
  | .beta | .dbeta | .retbeta => if f.vectorBeta then s.V else 1
  | _ => s.V
def planes (s : Shape) : Region → Nat
  | .dq | .dk | .dbeta => s.BNV*s.BH
  | _ => s.BH
def Valid (s : Shape) (f : Flags) (a : Addr) :=
  a.plane<planes s a.region ∧ a.row<rows s a.region ∧ a.col<cols s f a.region
def offset (s : Shape) (f : Flags) (a : Addr) :=
  (a.plane*rows s a.region+a.row)*cols s f a.region+a.col
def capacity (s : Shape) (f : Flags) (r : Region) := planes s r*rows s r*cols s f r
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Geometry
