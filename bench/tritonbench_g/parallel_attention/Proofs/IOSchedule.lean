import bench.tritonbench_g.parallel_attention.Proofs.IOExpected

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

@[simp] theorem forward_initial {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 128) (s : State α) :
    (forwardJob M D shape scale p).initial s=withProgram p s := rfl

@[simp] theorem query_initial {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State α) :
    (queryJob M D shape scale p).initial s=s := rfl

@[simp] theorem keyValue_initial {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State α) :
    (keyValueJob M D shape scale p).initial s=s := rfl

@[simp] theorem combined_initial (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State ℝ) :
    (combinedJob shape scale p).initial s=withProgram p s := rfl

theorem real_forward_schedule (shape : Shape) (scale : ℝ) (ps : List (Program shape 128)) (s : State ℝ) :
    runSchedule (ps.map (forwardJob (scalarAlgebra realScalars) (scalarDotPrimitive realScalars) shape scale)) s=
      runPrograms (forwardLaunch realScalars shape scale) ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.map_cons,runSchedule,runPrograms]
    change (forwardLaunch realScalars shape scale p s).bind
      (runSchedule (rest.map (forwardJob (scalarAlgebra realScalars) (scalarDotPrimitive realScalars) shape scale))) =
      (forwardLaunch realScalars shape scale p s).bind (runPrograms (forwardLaunch realScalars shape scale) rest)
    exact congrArg (Option.bind (forwardLaunch realScalars shape scale p s)) (funext ih)

theorem real_backward_schedule (shape : Shape) (scale : ℝ) (ps : List (Program shape 64)) (s : State ℝ) :
    runSchedule (ps.map (combinedJob shape scale)) s=
      runPrograms (combinedLaunch realScalars shape scale) ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.map_cons,runSchedule,runPrograms]
    change (combinedLaunch realScalars shape scale p s).bind
      (runSchedule (rest.map (combinedJob shape scale))) =
      (combinedLaunch realScalars shape scale p s).bind (runPrograms (combinedLaunch realScalars shape scale) rest)
    exact congrArg (Option.bind (combinedLaunch realScalars shape scale p s)) (funext ih)

#axiomsClean real_forward_schedule
#axiomsClean real_backward_schedule
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
