import bench.repairs.tritonbench_g.rotary_emb.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.RotaryEmb.Pair
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Reuse the established complete invocation equivalence, including both
safe source traces, all four native output channels and exterior frames. -/
theorem equivalent (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    RotaryIO.original rest p shape pids ≡[[]] RotaryIO.indexCommuted rest p shape pids :=
  ProgramIO.Implementation.of_equivalent (RotaryIO.candidate rest (Native.ops p) shape pids)
    (RotaryIO.equivalent rest (Native.ops p) shape valid pids)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RotaryEmb.Pair
