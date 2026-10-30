import bench.repairs.tritonbench_g.rotary_emb.Proofs.Pair

namespace VeriTile.Bench.Repairs.RotaryEmb.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original := RotaryIO.original
abbrev indexCommuted := RotaryIO.indexCommuted

/-- Compare the two actual complete FP32 invocations. Only integer head-block
multiplication is commuted; native mul/neg/FMA order and all observations stay
in the existing shared IO. Both source traces are safe, both executions succeed
and all four output channels and exterior frames agree on every native node. -/
specification rotary_emb_native_equiv (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    original rest p shape pids ≡[[]] indexCommuted rest p shape pids :=
  Pair.equivalent rest p shape valid pids

#stmtSurfaceSubset rotary_emb_native_equiv ⊆
  [original,indexCommuted,Spec.Equiv,Spec.RuleEntry,Spec.ProgramSyntax.Statement,
   ProgramIO.Inputs,ProgramIO.InputBindingMode.decoded,ProgramIO.Implementation,
   ProgramIO.Implementation.programSyntax,RotaryIO.kernelIO,Native.ops,
   Native.Node,Typed.instInhabitedNode,Native.Instructions,FP.Structural.Algebra,
   Shape,Shape.Valid]
#print_fp_assumptions rotary_emb_native_equiv
#axiomsClean rotary_emb_native_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.FPEquiv
