import VeriTile.Triton.DSL
namespace VeriTile.Tests.BooleanCallControlReference
open VeriTile.Triton

/-- Independent complete typed programs for all combinations of masked reads
and masked writes, including the unmasked meaning of a None mask. -/
def program (X Y : RegionName) (N B ITER : Nat) (readMasked writeMasked : Bool) : ComputeKernel :=
  match readMasked,writeMasked with
  | Bool.false,Bool.false => triton {
      tl.ports inputs=[X], outputs=[Y]
      tl.region X = tl.float32, Y = tl.float32
      lanes = tl.arange(0, $(B))
      x = tl.load(X + lanes)
      for «_» in range($(0), $(ITER), $(1)) {
        tl.store(Y + lanes, x)
      }
    }
  | Bool.false,Bool.true => triton {
      tl.ports inputs=[X], outputs=[Y]
      tl.region X = tl.float32, Y = tl.float32
      lanes = tl.arange(0, $(B))
      x = tl.load(X + lanes)
      mask_c = lanes < $(N)
      for «_» in range($(0), $(ITER), $(1)) {
        tl.store(Y + lanes, x, mask=mask_c)
      }
    }
  | Bool.true,Bool.false => triton {
      tl.ports inputs=[X], outputs=[Y]
      tl.region X = tl.float32, Y = tl.float32
      lanes = tl.arange(0, $(B))
      active = lanes < $(N)
      x = tl.load(X + lanes, mask=active, other=0.0)
      for «_» in range($(0), $(ITER), $(1)) {
        tl.store(Y + lanes, x)
      }
    }
  | Bool.true,Bool.true => triton {
      tl.ports inputs=[X], outputs=[Y]
      tl.region X = tl.float32, Y = tl.float32
      lanes = tl.arange(0, $(B))
      active = lanes < $(N)
      x = tl.load(X + lanes, mask=active, other=0.0)
      mask_c = lanes < $(N)
      for «_» in range($(0), $(ITER), $(1)) {
        tl.store(Y + lanes, x, mask=mask_c)
      }
    }
end VeriTile.Tests.BooleanCallControlReference
