import bench.tritonbench_g.relu_strided_buffer.Proofs.HostPlan
import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedGridExecution

/-!
Composition of actual wrapper preparation, its constexpr branch and the
concrete signed FP32 tile executor. This proves the host-model result and frame.
The runtime boundary is a successful serializable launch of these independent
tile effects. Native signed block-pointer lowering, finite-width indexing and
IEEE refinement are separate; no original-task acceptance is asserted here.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.HostExecution
open VeriTile.Triton SignedView SignedTile SignedGrid SignedGridExecution HostPlan

/-- Both production branches, with their actual pid-outer/step-inner schedule. -/
def sourceStarts (a : Arguments) : List Nat :=
  if a.oneTile then (List.range a.C).map (fun pid => pid * a.B)
  else (List.range a.C).flatMap fun pid =>
    (List.range a.T).map (fun step => (pid + step * a.C) * a.B)

noncomputable def executeArguments (a : Arguments) (s : BlockState) : Option BlockState :=
  runTiles (fun start state => stepStmts (tileBody a.input a.output a.n start a.B) state)
    (sourceStarts a) s

inductive Result where
  | hostFailure (failure : Failure) (state : BlockState)
  | executionFailure
  | returned (output : Buffer) (state : BlockState)

/-- The source returns the original `out0` object after execution. Preparation
failure preserves tensor memory. No callback supplies the output values. -/
noncomputable def execute (input output : Buffer) (guardSucceeds : Bool) (s : BlockState) : Result :=
  match prepare input output guardSucceeds with
  | .error failure => .hostFailure failure s
  | .ok a => match executeArguments a s with
    | none => .executionFailure
    | some final => .returned output final

theorem prepared_starts (input output : Buffer) (n : Nat) (sx sy : Int) :
    sourceStarts (rankOneArguments input output n sx sy) = cappedStarts n (block n) := by
  by_cases hone : CappedGrid.tilesPerCTA n (block n) = 1
  · simp only [sourceStarts, rankOneArguments, hone, decide_true,
      if_true]
    exact (cappedStarts_one_tile hone).symm
  · simp [sourceStarts, rankOneArguments, hone, cappedStarts]

theorem arguments_execution (input output : Buffer) (n : Nat) (sx sy : Int)
    (s : BlockState) :
    executeArguments (rankOneArguments input output n sx sy) s =
      runTiles (fun start state => stepStmts
        (tileBody (input.view sx) (output.view sy) n start (block n)) state)
        (cappedStarts n (block n)) s := by
  rw [executeArguments, prepared_starts]
  rfl

/-- Successful rank-one host-model execution, complete logical output from
the original input, exact preservation of input/exterior cells and bounds.
Signed strides of either sign are accepted under ordinary valid-view premises.
The only value premise binds input cells to the independent mathematical xs. -/
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
  obtain ⟨final, hexec, hresult, hinput, hframe⟩ :=
    fp32_capped_real_values (input.view sx) (output.view sy)
      hin hout hstride (block_pos hn) hsep (cappedStarts n (block n))
      (List.Perm.refl _) s xs hvalues
  refine ⟨final, ?_, hresult, hinput, hframe, ?_⟩
  · simp only [execute, rank_one_prepare input output hn sx sy
      hshapeIn hshapeOut hstrideIn hstrideOut,
      arguments_execution, hexec]
  · intro i hi
    exact ⟨address_lt _ hin hi, address_lt _ hout hi⟩

theorem failure_preserves_memory (input output : Buffer) (guardSucceeds : Bool)
    (s : BlockState) (failure : Failure)
    (h : prepare input output guardSucceeds = .error failure) :
    execute input output guardSucceeds s = .hostFailure failure s := by
  simp [execute, h]

theorem empty_failure (input output : Buffer) (guardSucceeds : Bool) (s : BlockState)
    (hin : input.shape = [0]) (hout : output.shape = [0]) :
    execute input output guardSucceeds s = .hostFailure .tileDivision s :=
  failure_preserves_memory input output guardSucceeds s .tileDivision
    (empty_before_guard input output guardSucceeds hin hout)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.HostExecution
