import bench.tritonbench_g.parallel_attention.Proofs.IOGraphSteps

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_stable (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) (s t : State α) (h : SameInputs s t) (R : RegionName) (off : Nat) :
    forwardTarget M D shape scale p s R off=forwardTarget M D shape scale p t R off := by
  simp only [forwardTarget,forward_values_inputs M D shape scale (withProgram p s) (withProgram p t) h rfl]

theorem query_stable (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (s t : State α) (h : SameInputs s t) (R : RegionName) (off : Nat) :
    queryTarget M D shape scale p s R off=queryTarget M D shape scale p t R off := by
  simp only [queryTarget,query_values_inputs M D shape scale p.tile s t h]

theorem keyValue_stable (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (s t : State α) (h : SameInputs s t) (R : RegionName) (off : Nat) :
    keyValueTarget M D shape scale p s R off=keyValueTarget M D shape scale p t R off := by
  simp only [keyValueTarget,keyValue_values_inputs M D shape scale p.tile s t h]

def backwardTarget (Q : Algebra α) (DQ : DotPrimitive α) (KV : Algebra α) (DKV : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State α) (R : RegionName) (off : Nat) : Cell α :=
  if R="dq" then queryTarget Q DQ shape scale p s R off else keyValueTarget KV DKV shape scale p s R off

def backwardRun (Q : Algebra α) (DQ : DotPrimitive α) (KV : Algebra α) (DKV : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State α) : Option (State α) :=
  ((queryJob Q DQ shape scale p).run s).bind (keyValueJob KV DKV shape scale p).run

theorem backward_step (Q : Algebra α) (DQ : DotPrimitive α) (KV : Algebra α) (DKV : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State α) :
    ∃ final,backwardRun Q DQ KV DKV shape scale p s=some final ∧
      (∀ R off,backwardWrites shape p R off → final.mem R off=backwardTarget Q DQ KV DKV shape scale p s R off) ∧
      (∀ R off,¬backwardWrites shape p R off → final.mem R off=s.mem R off) := by
  apply sequential_step _ _ ((queryRectangle shape p.tile).writes 64 shape.BK) (keyValueWrites shape p)
    (fun t=>queryTarget Q DQ shape scale p t) (fun t=>keyValueTarget KV DKV shape scale p t)
  · intro R off h
    exact h.1
  · intro R off h eq
    subst R
    rcases h with h|h <;> simpa [keyRectangle,queryRectangle,valueRectangle,outputRectangle] using h.1
  · intro R off member h
    have eq:=h.1
    change R="dq" at eq
    subst R
    simp at member
  · exact keyValue_stable KV DKV shape scale p
  · exact query_step Q DQ shape scale p
  · exact keyValue_step KV DKV shape scale p

theorem forward_grid (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (ps : List (Program shape 128)) (s : State α) :
    ∃ final,runPrograms (fun p=>(forwardJob M D shape scale p).run) ps s=some final ∧ SameInputs s final ∧
      (∀ p∈ps,∀ R off,forwardWrites shape p R off → final.mem R off=forwardTarget M D shape scale p s R off) ∧
      (∀ R off,(∀ p∈ps,¬forwardWrites shape p R off) → final.mem R off=s.mem R off) := by
  apply runPrograms_correct _ _ _ _ (forward_write_unique shape) (forward_stable M D shape scale)
    (forward_step M D shape scale)
  intro p R off member writes
  rcases writes with ⟨eq,_⟩|⟨_,eq,_⟩ <;> subst R <;> simp [outputRectangle] at member

theorem backward_grid (Q : Algebra α) (DQ : DotPrimitive α) (KV : Algebra α) (DKV : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (ps : List (Program shape 64)) (s : State α) :
    ∃ final,runPrograms (backwardRun Q DQ KV DKV shape scale) ps s=some final ∧ SameInputs s final ∧
      (∀ p∈ps,∀ R off,backwardWrites shape p R off → final.mem R off=backwardTarget Q DQ KV DKV shape scale p s R off) ∧
      (∀ R off,(∀ p∈ps,¬backwardWrites shape p R off) → final.mem R off=s.mem R off) := by
  apply runPrograms_correct _ _ _ _ (backward_write_unique shape) _ (backward_step Q DQ KV DKV shape scale)
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨eq,_⟩|⟨eq,_⟩ <;> subst R <;>
      simp [queryRectangle,keyRectangle,valueRectangle,outputRectangle] at member
  · intro p before after same R off
    simp only [backwardTarget,query_stable Q DQ shape scale p before after same,
      keyValue_stable KV DKV shape scale p before after same]

theorem forward_schedule (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (ps : List (Program shape 128)) (s : State α) :
    runSchedule (ps.map (forwardJob M D shape scale)) s=
      runPrograms (fun p=>(forwardJob M D shape scale p).run) ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.map_cons,runSchedule,runPrograms]
    exact congrArg (Option.bind ((forwardJob M D shape scale p).run s)) (funext ih)

theorem backward_schedule (Q : Algebra α) (DQ : DotPrimitive α) (KV : Algebra α) (DKV : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (ps : List (Program shape 64)) (s : State α) :
    runSchedule (ps.flatMap fun p=>[queryJob Q DQ shape scale p,keyValueJob KV DKV shape scale p]) s=
      runPrograms (backwardRun Q DQ KV DKV shape scale) ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.flatMap_cons,List.cons_append,List.nil_append,runSchedule,runPrograms,backwardRun]
    rw [Option.bind_assoc]
    congr 1
    funext t
    exact congrArg (Option.bind ((keyValueJob KV DKV shape scale p).run t)) (funext ih)

#axiomsClean forward_grid
#axiomsClean backward_grid
#axiomsClean forward_schedule
#axiomsClean backward_schedule
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOGraph
