import bench.tritonbench_g.relu_strided_buffer.Proofs.SourceLaunch
import bench.tritonbench_g.relu_strided_buffer.Proofs.SourceGridExecution
import bench.tritonbench_g.relu_strided_buffer.Proofs.HostExecution

/-!
The original wrapper's selected branch now executes its complete dynamic CTA
bodies, including the forRangeDyn loop, before returning the original out0.
The proven source execution refines the earlier concrete tile composition on
memory and unspecified padding. Native signed block-pointer lowering and
finite-width arithmetic remain separate from this mathematical typed executor.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceHostExecution
open VeriTile.Triton SignedView SignedTile SignedGrid HostPlan SourceLaunch

def views (a : Arguments) : SourceSignedBranch.SignedViews :=
  ⟨a.input, a.output, a.n, a.B⟩

def startsPerProgram (a : Arguments) (pid : Nat) : List Nat :=
  if a.oneTile then [pid * a.B]
  else (List.range a.T).map (fun j => (pid + j * a.C) * a.B)

theorem starts_flatten (a : Arguments) :
    (List.range a.C).flatMap (startsPerProgram a) = HostExecution.sourceStarts a := by
  change (List.range a.C).flatMap (fun pid => startsPerProgram a pid) = _
  cases h : a.oneTile with
  | «false» => simp only [startsPerProgram, HostExecution.sourceStarts, h, Bool.false_eq_true, if_false]
  | «true» =>
    simp only [startsPerProgram, HostExecution.sourceStarts, h, if_true]
    exact List.map_eq_flatMap.symm

noncomputable def executeArguments (a : Arguments) (s : BlockState) : Option BlockState :=
  runPrograms (SourceGridExecution.kernelBody (views a) a.oneTile a.T)
    a.C (List.range a.C) s

/-- The local source implementation premise is discharged by the proved
complete branch execution. No tile or kernel result callback remains. -/
theorem arguments_observations (a : Arguments) (s : BlockState) :
    ∃ actualFinal flatFinal,
      executeArguments a s = some actualFinal ∧
      HostExecution.executeArguments a s = some flatFinal ∧
      SourceBranch.ObsEq actualFinal flatFinal := by
  have hbody : ∀ pid state, ∃ actualFinal flatFinal,
      stepStmts (SourceGridExecution.kernelBody (views a) a.oneTile a.T)
        (atProgram a.C pid state) = some actualFinal ∧
      runTiles (fun start state => stepStmts (tileBody a.input a.output a.n start a.B) state)
        (startsPerProgram a pid) (atProgram a.C pid state) = some flatFinal ∧
      SourceBranch.ObsEq actualFinal flatFinal := by
    intro pid state
    simpa only [SourceGridExecution.executeKernel, SourceGridExecution.kernelStarts,
      views, atProgram, if_true, startsPerProgram] using
      SourceGridExecution.kernel_execution (views a) a.oneTile a.T (atProgram a.C pid state)
  obtain ⟨actualFinal, flatFinal, hactual, hflat, hobs⟩ :=
    programs_execution a.input a.output a.n a.B a.C
      (SourceGridExecution.kernelBody (views a) a.oneTile a.T)
      (startsPerProgram a) hbody (List.range a.C) s s (SourceBranch.ObsEq.refl s)
  refine ⟨actualFinal, flatFinal, hactual, ?_, hobs⟩
  simpa only [starts_flatten, HostExecution.executeArguments] using hflat

noncomputable def execute (input output : Buffer) (guardSucceeds : Bool)
    (s : BlockState) : HostExecution.Result :=
  match prepare input output guardSucceeds with
  | .error failure => .hostFailure failure s
  | .ok a => match executeArguments a s with
    | none => .executionFailure
    | some final => .returned output final

/-- Complete rank-one source-model execution: wrapper preparation, selected
source branch, all CTA loops, original output identity, logical max(0,x) and
exact input/exterior frame. The selected FP32 evaluator is WithBot Real, and
the runtime interpretation is serializable execution of independent CTAs. -/
theorem rank_one_success (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (hn : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity)
    (hstride : sy ≠ 0) (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (s : BlockState) (xs : Nat → ℝ)
    (hvalues : ∀ i, i < n → s.readMemValue .fp32 input.region
      ((input.view sx).address i) = some (xs i)) :
    ∃ final, execute input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        storedCell32 (some (max 0 (xs i)))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) := by
  obtain ⟨flat, hreturn, hresult, hinput, hframe, hbounds⟩ :=
    HostExecution.rank_one_success input output hn sx sy hshapeIn hshapeOut
      hstrideIn hstrideOut hin hout hstride hsep s xs hvalues
  obtain ⟨actual, flat', hactual, hflat, hobs⟩ :=
    arguments_observations (rankOneArguments input output n sx sy) s
  have hprepare := rank_one_prepare input output hn sx sy
    hshapeIn hshapeOut hstrideIn hstrideOut
  have hreturn' : HostExecution.execute input output Bool.true s = .returned output flat' := by
    simp only [HostExecution.execute, hprepare, hflat]
  have heq : flat = flat' := by
    cases hreturn.symm.trans hreturn'
    rfl
  subst flat'
  refine ⟨actual, ?_, ?_, ?_, ?_, hbounds⟩
  · simp only [execute, hprepare, hactual]
  · simpa only [hobs.1] using hresult
  · simpa only [hobs.1] using hinput
  · simpa only [hobs.1] using hframe

theorem failure_preserves_memory (input output : Buffer) (guardSucceeds : Bool)
    (s : BlockState) (failure : Failure)
    (h : prepare input output guardSucceeds = .error failure) :
    execute input output guardSucceeds s = .hostFailure failure s := by
  simp only [execute, h]

theorem empty_failure (input output : Buffer) (guardSucceeds : Bool) (s : BlockState)
    (hin : input.shape = [0]) (hout : output.shape = [0]) :
    execute input output guardSucceeds s = .hostFailure .tileDivision s :=
  failure_preserves_memory input output guardSucceeds s .tileDivision
    (empty_before_guard input output guardSucceeds hin hout)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceHostExecution
