import bench.tritonbench_g.sgmv_expand_slice.Proofs.IndexIO
import bench.tritonbench_g.sgmv_expand_slice.Proofs.IONative

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.FPEquiv
open VeriTile.Triton FP.Structural NativeBackend
open scoped VeriTile.Spec
noncomputable section

def original (rest : Algebra Raw) (ops : Native.Primitives) (a : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment) :=
  ReadProgramIO.Implementation.original
    (PublicIO.io (SourceBody.install (algebra rest ops (profile a)) (dot ops (weight a.weight)))
      a p shape segments) (PublicIO.nativeDomain p shape segments)

def indexReuse (rest : Algebra Raw) (ops : Native.Primitives) (a : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment) :=
  ReadProgramIO.Implementation.ofVariant
    (IndexPair.variant (SourceBody.install (algebra rest ops (profile a)) (dot ops (weight a.weight)))
      a p shape segments) (PublicIO.nativeDomain p shape segments)

/-- Complete original selected FP32/FP16/FP32 source versus actual tile-index
reuse candidate. Full K loops, eight metadata ABIs, ordered overlapping segments,
all result reads and exterior cells are retained. No floating rewrite is used. -/
specification sgmv_grid_equiv (rest : Algebra Raw) (ops : Native.Primitives)
    (a : SourceABI) (hi : a.input=.single) (hw : a.weight=.half) (ho : a.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p) (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    original rest ops a p shape segments ≡[[]] indexReuse rest ops a p shape segments := by
  exact ReadProgramIO.Implementation.of_equivalentOn
    (IndexPair.variant (SourceBody.install (algebra rest ops (profile a)) (dot ops (weight a.weight)))
      a p shape segments)
    (IndexPair.equivalent (algebra rest ops (profile a)) (dot ops (weight a.weight))
      a hi hw ho p shape segments separation fits inner columns slice capacity layout
      (PublicIO.nativeDomain p shape segments) (fun _ h => h.1))

#axiomsClean sgmv_grid_equiv
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.FPEquiv
