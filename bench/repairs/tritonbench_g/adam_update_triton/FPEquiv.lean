import bench.repairs.tritonbench_g.adam_update_triton.Proofs.WholePair

/-! Actual original/candidate equivalence for the existing momentum-addition
commutation. Both full source schedules execute with runtime FP32 arguments.
The view fixes the source-selected contracted native graph; the experiment
rule set is empty. This is a model-specific structural result on all words. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Spec
noncomputable section

/-- The original complete quoted source, in the actual full-grid schedule. -/
def original (native : NativeWords.Instructions) (params : Scalars Native.Word)
    (total n B : Nat) (pids : List Nat) :=
  ProgramIO.Implementation.original
    (WholePair.programIO .original native params total n B pids) (fun _ => True)

/-- The existing candidate changes `diff*beta2+grad` to `grad+diff*beta2`.
Its four runtime scalar arguments, mask, stores, grid and native ports agree. -/
def candidate (native : NativeWords.Instructions) (params : Scalars Native.Word)
    (total n B : Nat) (pids : List Nat) :=
  ProgramIO.Implementation.ofVariant
    (io := WholePair.programIO .original native params total n B pids)
    { schedule := (WholePair.programIO .commuted native params total n B pids).schedule
      observe := (WholePair.programIO .commuted native params total n B pids).observe }
    (fun _ => True)

/-- Both complete programs are safe and terminate, every active typed P/M cell
has the same native word, and each complete exterior memory frame is retained.
Every FP32 bit pattern and every permutation of the complete grid is admitted. -/
specification lion_kernel_equiv (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B : Nat) (positive : 0 < B)
    (pids : List Nat) (complete : pids.Perm (List.range ((n+B-1)/B))) :
    original native params total n B pids ≡[[]] candidate native params total n B pids := by
  apply ProgramIO.Implementation.of_equivalent
  exact ProgramIO.EquivalentOn.of_implementsOn
    (WholePair.implements .original native params total n B positive pids complete)
    (WholePair.implements .commuted native params total n B positive pids complete)

#axiomsClean lion_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.FPEquiv
