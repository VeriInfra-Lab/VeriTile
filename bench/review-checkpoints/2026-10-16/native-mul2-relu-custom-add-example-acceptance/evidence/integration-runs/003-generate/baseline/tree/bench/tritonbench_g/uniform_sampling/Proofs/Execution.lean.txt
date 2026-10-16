import bench.tritonbench_g.uniform_sampling.Proofs.ValueExecution
import bench.tritonbench_g.uniform_sampling.Proofs.OffsetExecution
import bench.tritonbench_g.uniform_sampling.Proofs.StorePhase

/-! Complete execution of all twenty-three statements of the checked source
projection. The result uses the independent scalar and four-store memory
specification. The public source lemma retains both original signed64 ABI
conditions and the launched signed-i32 index domain.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Execution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def resultBase (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState) : BlockState :=
  OffsetExecution.offsetState (u.pids 0) B (ValueExecution.valueState seed offset lower upper B u)

theorem block_register (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState) :
    (ValueExecution.valueState seed offset lower upper B u).regs .nat [] "BLOCK_arg" =
      some (Tile.scalar B) := by
  simp [ValueExecution.valueState, ValueExecution.affineState, ValueExecution.affineAssignState,
    ValueExecution.randomState, TupleExecution.outputState, TupleExecution.fieldState,
    TupleExecution.fieldName, TupleExecution.capturedState, IndexExecution.preparedState,
    IndexExecution.wrappedState, IndexExecution.indexState, CounterExecution.counterState,
    CounterExecution.lowState, CounterExecution.boundState]

theorem result_values (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState)
    (f : Fin 4) :
    (resultBase seed offset lower upper B u).regs .real [B] (StorePhase.wordName f) =
      some ⟨fun i => some (ScalarSpec.value seed offset
        (SourceDomain.counterIndex u.pid B i.1.val) (Formula.field f.val) lower upper)⟩ := by
  have hv := ValueExecution.value_register seed offset lower upper B u (Formula.field f.val)
  fin_cases f
  all_goals simpa [resultBase, OffsetExecution.offsetState, StorePhase.wordName,
    TupleExecution.fieldName, Formula.field, IndexExecution.indices, BlockState.pid] using hv

theorem result_offsets (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState)
    (f : Fin 4) :
    (resultBase seed offset lower upper B u).regs .nat [B] (StorePhase.offsetName f) =
      some ⟨fun i => SourceDomain.outputIndex u.pid B f i.1⟩ := by
  fin_cases f
  all_goals simp [resultBase, OffsetExecution.offsetState, StorePhase.offsetName,
    OffsetExecution.outputFn, SourceDomain.outputIndex, BlockState.pid]
  all_goals rfl

theorem offset_phase_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState) :
    stepStmts (((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 15).take 4)
      (ValueExecution.valueState seed offset lower upper B u) =
        some (resultBase seed offset lower upper B u) := by
  rw [OffsetExecution.source_slice]
  have he := OffsetExecution.offsets_exec B (ValueExecution.valueState seed offset lower upper B u)
    (block_register seed offset lower upper B u)
  rw [(ValueExecution.valueState_frame seed offset lower upper B u).2.1] at he
  exact he

theorem prefix_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    stepStmts ((Algorithm.kernel swapped y N seed offset lower upper B).body.take 19) u =
      some (resultBase seed offset lower upper B u) := by
  change stepStmts
    (((Algorithm.kernel swapped y N seed offset lower upper B).body.take 15) ++
      ((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 15).take 4) u = _
  rw [CounterExecution.steps_append, ValueExecution.prefix_exec swapped y N seed offset lower upper B u hseed,
    Option.bind_some, offset_phase_exec]

theorem resultBase_frame (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState) :
    (resultBase seed offset lower upper B u).mem = u.mem ∧
    (resultBase seed offset lower upper B u).pids = u.pids ∧
    (resultBase seed offset lower upper B u).undef = u.undef ∧
    (resultBase seed offset lower upper B u).numPids = u.numPids := by
  have ho := OffsetExecution.offsetState_frame (u.pids 0) B
    (ValueExecution.valueState seed offset lower upper B u)
  have hv := ValueExecution.valueState_frame seed offset lower upper B u
  exact ⟨ho.1.trans hv.1, ho.2.1.trans hv.2.1,
    ho.2.2.1.trans hv.2.2.1, ho.2.2.2.trans hv.2.2.2⟩

theorem exec_storeBlock (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : 0 < B) (u : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    exec (Algorithm.kernel swapped y N seed offset lower upper B) u =
      some (Memory.storeBlock u.pid (resultBase seed offset lower upper B u)
        y N B seed offset lower upper) := by
  unfold exec
  conv_lhs => rw [← List.take_append_drop 19 (Algorithm.kernel swapped y N seed offset lower upper B).body]
  rw [CounterExecution.steps_append, prefix_exec swapped y N seed offset lower upper B u hseed,
    Option.bind_some, StorePhase.source_slice]
  exact StorePhase.stores_exec hB u.pid y N seed offset lower upper _
    (result_values seed offset lower upper B u) (result_offsets seed offset lower upper B u)

theorem source_exec (storage : TypedSource.Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (_hN : N < 2^31) (_hpid : u.pid < SourceDomain.grid N B)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (_hoffset : Random.IntegerType.i64.inRange offset = Bool.true) :
    exec (TypedSource.source storage swapped y N seed offset lower upper B hB).eraseDType.toAlgKernel u =
      some (Memory.storeBlock u.pid (resultBase seed offset lower upper B u)
        y N B seed offset lower upper) := by
  simp only [ComputeKernel.toAlgKernel, Erasure.complete_projection, Algorithm.source_eq]
  exact exec_storeBlock swapped y N seed offset lower upper B (SourceDomain.legalBlock_fields hB).1 u hseed

#axiomsClean block_register
#axiomsClean result_values
#axiomsClean result_offsets
#axiomsClean offset_phase_exec
#axiomsClean prefix_exec
#axiomsClean resultBase_frame
#axiomsClean exec_storeBlock
#axiomsClean source_exec

end
end VeriTile.Bench.TritonBenchG.UniformSampling.Execution
