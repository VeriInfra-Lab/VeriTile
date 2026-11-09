import bench.tritonbench_g.fast_rope_embedding.IO
import VeriTile.Meta.StatementAudit

/-! Bind both shared IO descriptions to their own complete Python quotation.
The ordinary kernel receives its original default constexpr group size; the
fast kernel retains its source assignment. No equality of the two ASTs is assumed. -/
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.InvocationBindings
open VeriTile.Triton
open VeriTile.Bench.Repairs.RopeEmbedding

theorem fast_source (op : Operations A) (r : Request) (QR CR SR : Nat)
    (back : Bool) (row group : Nat) :
    (Interface.invocation op r QR CR SR back row group).kernel =
      Kernels.rope "Q" "C" "S" QR CR SR r.sequence r.dimension r.heads r.block back := rfl

theorem ordinary_source (op : Operations A) (r : Request) (QR CR SR : Nat)
    (back : Bool) (row group : Nat) :
    (Interface.invocation op r QR CR SR back row group Bool.false).kernel =
      RopeEmbedding.Kernels.ropeKernel "Q" "C" "S" QR CR SR r.sequence r.dimension r.heads r.block back := rfl

#axiomsClean fast_source
#axiomsClean ordinary_source
end VeriTile.Bench.Repairs.FastRopeEmbedding.InvocationBindings
