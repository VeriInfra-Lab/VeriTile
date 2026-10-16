import bench.tritonbench_g.vector_addition.Proofs.NativeArithmetic
import bench.tritonbench_g.add_value.Proofs.NativeScalar

namespace AddValueNative
open VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost Finite
abbrev hostExtent := Native.hostExtent
abbrev grid := Native.grid
abbrev gridSupported := Native.gridSupported
abbrev bound := Native.bound
abbrev offset := Native.offset
abbrev mask := Native.mask
abbrev pointer := Native.pointer
abbrev activeTile := Native.activeTile
abbrev Buffer := Native.Buffer
abbrev dtype (d : Storage) : DType := wideDType d
structure Layout where
  x : Buffer
  output : Buffer
  device : Nat
def Layout.Fresh (p : Layout) : Prop := p.x.Disjoint p.output

theorem all_masks_false_at_two31 (pid lane : Nat) : mask (2^31) pid lane=Bool.false := Native.all_masks_false_at_two31 pid lane
theorem active_tile_empty_at_two31 (pid : Nat) : activeTile (2^31) pid=[] := Native.active_tile_empty_at_two31 pid
theorem counterexample_grid : grid (2^31)=2^21 ∧ gridSupported (2^31) ∧ hostExtent (2^31) := Native.counterexample_grid
#axiomsClean all_masks_false_at_two31
#axiomsClean counterexample_grid
end AddValueNative
