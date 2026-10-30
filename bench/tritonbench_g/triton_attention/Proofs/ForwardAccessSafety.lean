import bench.tritonbench_g.triton_attention.Proofs.ForwardAccessPrologue
import bench.tritonbench_g.triton_attention.Proofs.ForwardAccessBody
import bench.tritonbench_g.triton_attention.Proofs.ForwardAccessEpilogue
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000

theorem forward_loop_safe (rest : Algebra α) (P : Primitives α)
    (p : ForwardParameters) (initial current : State α) (hn : 0<p.blockN)
    (hd : p.blockN∣forwardStop p initial) (block : Nat)
    (inv : ForwardInvariant P p initial block (forwardRounds P p initial block) current)
    (bounds : RegionName→Nat)
    (keys : ∀ n,n<forwardBlocks p initial → ∀ i : FKey p,forwardKVOffset p initial n i<bounds p.k)
    (values : ∀ n,n<forwardBlocks p initial → ∀ i : FKey p,forwardKVOffset p initial n i<bounds p.v) :
    Safety.rangeSafe (backend rest P) bounds "start_n" (block*p.blockN)
      (forwardStop p initial) p.blockN (forwardBody p) current := by
  have stop : forwardStop p initial=forwardBlocks p initial*p.blockN := (Nat.div_mul_cancel hd).symm
  apply Safety.rangeSafe_of_invariant (backend rest P) bounds "start_n" (forwardStop p initial) p.blockN
    (forwardBody p) (fun offset s=>∃ n,offset=n*p.blockN ∧
      ForwardInvariant P p initial n (forwardRounds P p initial n) s) _ _ _ _ ⟨block,rfl,inv⟩
  · intro offset active s invariant
    obtain ⟨n,rfl,h⟩ := invariant
    have live : n<forwardBlocks p initial := by rw [stop] at active;exact (Nat.mul_lt_mul_right hn).mp active
    exact forward_body_safe rest P p initial s n _ hn h bounds (keys n live) (values n live)
  · intro offset active s invariant t ran
    obtain ⟨n,rfl,h⟩ := invariant
    rw [forwardBodyExecution rest P p initial s n _ hn h] at ran
    cases ran
    refine ⟨n+1,by simp [Nat.add_mul],?_⟩
    exact forwardBodyInvariant P p initial s n _ h

/-- The complete original forward quotation has backed accesses for every
legal CTA, for either Real or compiler-staged numerical interpretation. -/
theorem forward_safe (rest : Algebra α) (P : Primitives α)
    (c : Shape) (scale : ℝ) (initial : State α)
    (legal : initial.pids 0<c.blocks ∧ initial.pids 1<c.groups)
    (bounds : RegionName→Nat) (backed : Bounds c bounds) :
    TraceSafe (backend rest P) bounds (forwardKernel c scale) initial := by
  let p := forwardParameters c scale
  have query : ∀ i : FQuery p,forwardOffset p initial i<bounds "Q" :=
    fun i=>lt_of_lt_of_le (forward_query_bound c scale initial legal i) (backed.1 _ (by simp))
  have keys : ∀ n,n<forwardBlocks p initial → ∀ i : FKey p,forwardKVOffset p initial n i<bounds "K" :=
    fun n hn i=>lt_of_lt_of_le (forward_kv_bound c scale initial legal n hn i) (backed.1 _ (by simp))
  have values : ∀ n,n<forwardBlocks p initial → ∀ i : FKey p,forwardKVOffset p initial n i<bounds "V" :=
    fun n hn i=>lt_of_lt_of_le (forward_kv_bound c scale initial legal n hn i) (backed.1 _ (by simp))
  have hn : 0<p.blockN := by change 0<128; decide
  have hd : p.blockN∣forwardStop p initial := dvd_mul_left _ _
  let start := forwardPrologueResult P p initial
  have startInv := forwardPrologueInvariant P p initial
  have safeLoop := forward_loop_safe rest P p initial start hn hd 0 startInv bounds keys values
  obtain ⟨middle,loopRan,middleInv⟩ := forwardLoopExecution rest P p initial start 0 (by omega) hn hd startInv
  have loopRun : run (backend rest P)
      [.forRangeDyn "start_n" (.constNat 0)
        (.mul .nat .nil (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat p.blockM))
        (.constNat p.blockN) (forwardBody p)] start=some middle := by
    simpa [run,step,evalOp_unfold,numeric,startInv.start,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,forwardStop] using loopRan
  change Safety.traceSafeList _ _ (forwardSource p).surfaceBody initial
  rw [forwardDecomposition,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨forward_prologue_safe rest P p initial bounds query,?_⟩,?_⟩
  · intro next ran
    rw [forwardPrologueExecution rest P p initial] at ran
    cases ran
    simpa [AccessSafety.singleton_safe,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,startInv.start,Broadcast.leftIndex,Broadcast.rightIndex,forwardStop]
      using safeLoop
  · intro next ran
    rw [run_append,forwardPrologueExecution rest P p initial] at ran
    simp only [Option.bind_some] at ran
    have equal : middle=next := Option.some.inj (loopRun.symm.trans ran)
    subst next
    apply forward_epilogue_safe rest P p initial middle _ _ middleInv bounds
    · intro i
      exact lt_of_lt_of_le (forward_row_bound c scale initial legal i) (backed.2 "L" (by simp))
    · intro i
      exact lt_of_lt_of_le (forward_row_bound c scale initial legal i) (backed.2 "M" (by simp))
    · intro i
      exact lt_of_lt_of_le (forward_query_bound c scale initial legal i) (backed.1 "Out" (by simp))

#axiomsClean forward_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
