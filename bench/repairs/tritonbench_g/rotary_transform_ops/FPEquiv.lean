import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.XorIO

namespace VeriTile.Bench.Repairs.RotaryTransformOps.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Native.Node := ⟨Native.atom Native.zero⟩

def original (c : Config) (p : Native.Instructions) (cu64 off64 : Bool) :=
  ReadProgramIO.Implementation.original
    (Interface.io (Native.ops p) c cu64 off64 Interface.observeNative) (Interface.Ready c)

def xorIndexed (c : Config) (p : Native.Instructions) (cu64 off64 : Bool) :=
  ReadProgramIO.Implementation.ofVariant
    (XorPair.variant (Native.ops p) c cu64 off64 Interface.observeNative) (Interface.Ready c)

/-- Exact adjacent-partner index optimization of the complete original grid.
Both actual sources execute safely and successfully, agree on all result reads
including the copied suffix, and preserve both exterior frames. Source FP32
arithmetic, metadata widths and every raw input word remain unchanged. -/
specification rotary_grid_equiv (c : Config) (valid : c.Valid)
    (p : Native.Instructions) (cu64 off64 : Bool) :
    original c p cu64 off64 ≡[[]] xorIndexed c p cu64 off64 := by
  exact ReadProgramIO.Implementation.of_equivalentOn
    (XorPair.variant (Native.ops p) c cu64 off64 Interface.observeNative)
    (XorPair.equivalent c valid p cu64 off64)

#axiomsClean rotary_grid_equiv
end
end VeriTile.Bench.Repairs.RotaryTransformOps.FPEquiv
