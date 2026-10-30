import bench.tritonbench_g.layer_norm_triton.ForwardPairIO

import bench.tritonbench_g.layer_norm_triton.BackwardPairIO

/-! Actual two-program equivalence for every retained source-stage scope.
The existing forward loop expansion and two backward lifetime schedules retain
all FP32 operations, casts, masks, outputs and exterior memory frames. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Triton FP.Structural SourceModel ForwardPairIO
open scoped VeriTile.Spec
noncomputable section

/-- Full source/candidate observations, with every active output and raw frame.
This is the existing control-flow transformation, for arbitrary FP semantics. -/
specification forward_equiv {α : Type} [Inhabited α] (m : Model α)
    (stride N B row : Nat) (eps : ℝ) (positive : 0<N) (covers : N≤B) :
    original m stride N B row eps ≡[[]] candidate m stride N B row eps :=
  ProgramIO.Implementation.of_equivalent
    (ForwardEquivalenceIO.candidate m stride N B row eps (fun _ _ => id))
    (ForwardEquivalenceIO.equivalent m stride N B row eps (fun _ _ => id) positive covers)

/-- Grouped backward retains the complete row loop, every DX result read and
both partial sums; all-word scalar meanings remain arbitrary. -/
specification grouped_equiv {α : Type} [Inhabited α] (m : Model α)
    (stride M N G B group : Nat) :
    BackwardPairIO.Grouped.original m stride M N G B group ≡[[]]
      BackwardPairIO.Grouped.candidate m stride M N G B group :=
  ReadProgramIO.Implementation.of_equivalent
    (BackwardPairIO.Grouped.variant m stride M N G B group)
    (BackwardPairIO.Grouped.equivalent m stride M N G B group)

/-- Final backward retains the complete partial-row reduction and both stores. -/
specification final_equiv {α : Type} [Inhabited α] (m : Model α)
    (M N BM BN pid : Nat) :
    BackwardPairIO.Final.original m M N BM BN pid ≡[[]]
      BackwardPairIO.Final.candidate m M N BM BN pid :=
  ProgramIO.Implementation.of_equivalent
    (BackwardPairIO.Final.variant m M N BM BN pid)
    (BackwardPairIO.Final.equivalent m M N BM BN pid)

#axiomsClean grouped_equiv
#axiomsClean final_equiv

#print_fp_assumptions forward_equiv
#stmtSurfaceSubset forward_equiv ⊆
  [RegionName,Model,original,candidate,Spec.Equiv,ProgramIO.Implementation,Interface.Forward.io,
   ProgramIO.Inputs,Spec.RuleEntry,Spec.ProgramSyntax.Statement,
   ProgramIO.Implementation.programSyntax,ProgramIO.InputBindingMode.decoded]
#axiomsClean forward_equiv
end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
