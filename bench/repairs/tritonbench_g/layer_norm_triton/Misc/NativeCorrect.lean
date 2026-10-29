import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.GroupedIOResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.FinalIO
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.NativeObservationDefs
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardEquivalenceIO

namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Triton FP.Structural SourceModel SourceNative SourceNative.Native Interface
noncomputable section

/-- Complete grouped source behavior includes zero-step loops, overlapping
rows, both partial-gradient arrays and every DX readback. -/
theorem grouped_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (stride M N G B group : Nat) :
    (Grouped.io (model R) stride M N G B group (observeWords I S)).Implements
      (GroupedReference.expected (model R) stride M N G B group (observeWords I S)) :=
  GroupedIOResult.implements (model R) stride M N G B group (observeWords I S)

/-- All final partial-gradient rows, both outputs, exact original BM/BN scope. -/
theorem final_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (M N BM BN pid : Nat) :
    (Final.io (model R) M N BM BN pid (observeWords I S)).Implements
      (FinalIO.expected (model R) M N BM BN pid (observeWords I S)) :=
  FinalIO.implements (model R) M N BM BN pid (observeWords I S)

/-- The complete raw IO result underlying the forward program pair. -/
theorem forward_io_equiv {α : Type} [Inhabited α] (m : Model α)
    (stride N B row : Nat) (eps : ℝ) (positive : 0<N) (covers : N≤B) :
    (Forward.io m stride N B row eps (fun _ _ => id)).Equivalent
      (ForwardEquivalenceIO.candidate m stride N B row eps (fun _ _ => id)) :=
  ForwardEquivalenceIO.equivalent m stride N B row eps (fun _ _ => id) positive covers

#specNonCircular GroupedReference.expected avoiding [FP.Structural.exec,FP.Structural.run,TypedSource.groupedBackward,SourceGroupedExecution.final]
#specNonCircular FinalIO.expected avoiding [FP.Structural.exec,FP.Structural.run,TypedSource.finalBackward,SourceFinalGraph.final]
#stmtSurfaceSubset grouped_native ⊆ [Term,Reductions,Instructions,OutputSchedule,Grouped.io,model,observeWords,
  ReadProgramIO.Implements,GroupedReference.expected,FastLayernorm.KernelNative.instInhabitedTerm]
#stmtSurfaceSubset final_native ⊆ [Term,Reductions,Instructions,OutputSchedule,Final.io,model,observeWords,
  ProgramIO.Implements,FinalIO.expected,FastLayernorm.KernelNative.instInhabitedTerm]
#axiomsClean grouped_native
#axiomsClean final_native
#stmtSurfaceSubset forward_io_equiv ⊆ [RegionName,Model,Forward.io,ProgramIO.Equivalent,ForwardEquivalenceIO.candidate]
#axiomsClean forward_io_equiv
end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
