import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.WhileBody
open VeriTile.Triton
noncomputable section

def counterSignature : CallSignature where
  scalars := [("limit", .integer)]

/-- The literal Python loop mutates the register read by its guard. -/
def counter (limit : Nat) : ComputeKernel :=
  with_triton_call counterSignature call_args [limit := limit] in
  triton {
    limit64 = tl.cast(limit, tl.uint64)
    i = tl.full((), 0, tl.uint64)
    while i < limit64:
        i += 1
  }

theorem retains_while (limit : Nat) :
    (match (counter limit).surfaceBody[2]? with
      | some (.whileLoop _ [_]) => Bool.true
      | _ => Bool.false) = Bool.true := rfl

def wrapping : ComputeKernel :=
  with_triton_call counterSignature call_args [limit := (0 : Nat)] in
  triton {
    i = tl.full((), 18446744073709551615, tl.uint64)
    zero = tl.full((), 0, tl.uint64)
    while i > zero:
        i += 1
  }

-- Inspect the elaborated source before fixing its execution theorem.

end
end VeriTile.Tests.WhileBody
