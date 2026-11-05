import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Single-output IO shared by this case's five kernels. The index type can
be a matrix lane or a streamed column/lane pair. Specifications supply their
own input arrays, grid, output addresses, masks and active-access bounds.
No final memory or output equation is assumed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
open VeriTile.Triton
noncomputable section

structure IO (Grid Input Index : Type) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  grid : BlockState → Grid
  read : BlockState → Input
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  out : RegionName
  write : Grid → Index → Nat
  mask : Grid → Index → Prop

def IO.writes {G I X : Type} (io : IO G I X) (A : FlatAlloc) (g : G) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ i, io.mask g i ∧ cell.2 = A.addr io.out (io.write g i)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
