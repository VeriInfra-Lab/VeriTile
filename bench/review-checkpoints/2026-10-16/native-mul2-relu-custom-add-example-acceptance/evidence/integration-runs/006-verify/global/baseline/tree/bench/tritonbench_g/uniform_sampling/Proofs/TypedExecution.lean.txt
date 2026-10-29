import bench.tritonbench_g.uniform_sampling.Proofs.TypedValues
import bench.tritonbench_g.uniform_sampling.Proofs.TypedStores
import bench.tritonbench_g.uniform_sampling.Proofs.FP
import VeriTile.Triton.Float.ProfiledRewrite

/-! Successful complete typed execution for every floating algebra and every
initial state in the original signed64 argument domain. The explicit witness
uses the wrapper's B=1024, N=4095, negative seed and offset: each of the four
streams writes active lanes, the last lane is masked, and the low counter
wraps. The canonical generator is evaluated symbolically, never precomputed.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedExecution
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

theorem source_executes {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) (u : State α)
    (_hN : N < 2^31) (_hpid : u.pids 0 < SourceDomain.grid N B)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (_hoffset : Random.IntegerType.i64.inRange offset = Bool.true) :
    ∃ v, FP.Structural.exec M (source storage swapped y N seed offset lower upper B hB) u = some v := by
  have ho := TypedValues.result_offsets M seed offset lower upper B u
  obtain ⟨v, hv⟩ := TypedStores.stores_exec M storage y N B hB
    (TypedValues.result M seed offset lower upper B u) _ _ _ _ _ _ _ _
    (TypedValues.result_values M seed offset lower upper B u .c0)
    (TypedValues.result_values M seed offset lower upper B u .c1)
    (TypedValues.result_values M seed offset lower upper B u .c2)
    (TypedValues.result_values M seed offset lower upper B u .c3)
    ho.1 ho.2.1 ho.2.2.1 ho.2.2.2
  refine ⟨v, ?_⟩
  change run M (source storage swapped y N seed offset lower upper B hB).surfaceBody u = some v
  rw [← List.take_append_drop 19 (source storage swapped y N seed offset lower upper B hB).surfaceBody,
    run_append, TypedValues.prefix_exec M storage swapped y N seed offset lower upper B hB u hseed,
    Option.bind_some, TypedStores.source_slice]
  exact hv

/-- A consistency model with ordinary real arithmetic and the actual finite
fp32 bit decoder. It witnesses satisfiability of the same public FP engine;
it is not an assertion that identity casts implement device rounding. -/
def consistencyModel : Algebra ℝ where
  literal := fun _ _ value => value
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b
    | .max => max a b | .pow => 0
  unary := fun _ _ value => value
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  cast := fun _ _ _ value => value
  fromNat := fun _ value => value
  fromInt := fun _ value => value
  fp32Bits := fun value => (value.decodeRat.getD 0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def initial : State ℝ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem block_legal : SourceDomain.LegalBlock 1024 := ⟨10, by decide, by decide⟩

theorem witness_domain : 4095 < 2^31 ∧ 0 < SourceDomain.grid 4095 1024 ∧
    Random.IntegerType.i64.inRange (-1) = Bool.true := by decide

theorem all_four_streams_active (f : Fin 4) :
    SourceDomain.outputIndex 0 1024 f ⟨0, by decide⟩ < 4095 := by
  fin_cases f <;> decide

theorem tail_masked :
    ¬ SourceDomain.outputIndex 0 1024 3 ⟨1023, by decide⟩ < 4095 := by decide

theorem counter_wrap : ScalarSpec.counterLow (-1) 1 = 0 := by decide

/-- A common successful final state in exactly the engine exported by
FPEquiv. Initial registers are empty and all four output streams are active.
The tail-mask and negative-offset wrap facts above describe this same run. -/
theorem nonempty_pair (storage : Storage) :
    ∃ v,
      FP.Structural.exec (FP.ProfiledRewrite.engine consistencyModel)
        (source storage Bool.false "out" 4095 (-1) (-1) 0 1 1024 block_legal) initial = some v ∧
      FP.Structural.exec (FP.ProfiledRewrite.engine consistencyModel)
        (source storage Bool.true "out" 4095 (-1) (-1) 0 1 1024 block_legal) initial = some v := by
  obtain ⟨v, he⟩ := source_executes (FP.ProfiledRewrite.engine consistencyModel)
    storage Bool.false "out" 4095 (-1) (-1) 0 1 1024 block_legal initial
    witness_domain.1 witness_domain.2.1 witness_domain.2.2 witness_domain.2.2
  exact ⟨v, he, (FPProof.whole_state storage "out" 4095 (-1) (-1) 0 1 1024 block_legal
    (FP.ProfiledRewrite.engine consistencyModel) initial).symm.trans he⟩

#axiomsClean source_executes
#axiomsClean block_legal
#axiomsClean witness_domain
#axiomsClean all_four_streams_active
#axiomsClean tail_masked
#axiomsClean counter_wrap
#axiomsClean nonempty_pair

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedExecution
