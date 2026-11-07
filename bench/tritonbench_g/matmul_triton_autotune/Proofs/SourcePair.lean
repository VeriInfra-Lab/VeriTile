import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_FPTypedSource

/-! The two complete pasted kernels differ only by commuting the integer
product that counts programs in a group. This preserves the full state, with
no floating-point law, launch restriction, or successful-execution premise. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.SourcePair
open VeriTile.Triton FP.Structural
open FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def before (p : Parameters) := (FPValue.source p .single).surfaceBody.take 12
def after (p : Parameters) := (FPValue.source p .single).surfaceBody.drop 13
def originalIndex (p : Parameters) :=
  ((FPValue.source p .single).surfaceBody[12]?).getD (.effectMarker "missing index")
def candidateIndex (p : Parameters) :=
  ((Source.kernel p).surfaceBody[12]?).getD (.effectMarker "missing index")

theorem original_split (p : Parameters) :
    (FPValue.source p .single).surfaceBody=before p++[originalIndex p]++after p := rfl

theorem candidate_split (p : Parameters) :
    (Source.kernel p).surfaceBody=before p++[candidateIndex p]++after p := rfl

theorem index_equal {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    step M (originalIndex p) s=step M (candidateIndex p) s := by
  cases hr : s.regs .int [] "num_pid_n" <;>
    simp [originalIndex,candidateIndex,FPValue.source,FPValue.typed,Source.kernel,Source.typed,
      Pasted.originalKernel,Pasted.candidateKernel,ComputeKernel.surfaceBody,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,hr,Int.mul_comm]
  all_goals apply congrArg (s.setReg "num_pid_in_group" .int [])
  all_goals funext i
  all_goals exact Int.mul_comm _ _

theorem execution_equal {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    FP.Structural.exec M (FPValue.source p .single) s=
      FP.Structural.exec M (Source.kernel p) s := by
  change run M (FPValue.source p .single).surfaceBody s=run M (Source.kernel p).surfaceBody s
  rw [original_split,candidate_split,run_append,run_append,run_append,run_append]
  apply congrArg (fun z : Option (State α) => z.bind (run M (after p)))
  apply congrArg (fun f => (run M (before p) s).bind f)
  funext t
  simp only [run,Option.bind_some,index_equal]

/-- The whole prefix establishes exactly the same registers in both programs. -/
theorem prelude_execution_equal {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    run M (FPValue.prelude p .single) s=run M ((Source.kernel p).surfaceBody.take 22) s := by
  let middle := ((FPValue.source p .single).surfaceBody.drop 13).take 9
  have ho : FPValue.prelude p .single=before p++[originalIndex p]++middle := rfl
  have hc : (Source.kernel p).surfaceBody.take 22=before p++[candidateIndex p]++middle := rfl
  have hi : run M [originalIndex p]=run M [candidateIndex p] := by
    funext t
    simp only [run,index_equal]
  rw [ho,hc,run_append,run_append,run_append,run_append,hi]

#axiomsClean prelude_execution_equal

#axiomsClean index_equal
#axiomsClean execution_equal
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.SourcePair
