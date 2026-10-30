import bench.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceSafety
open VeriTile.Triton FP.Structural SourceValue
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
private theorem clamped_lt (n j : Nat) : (if j<n-1 then j else n-1)<max n 1 := by
  split_ifs <;> omega

/-- All potentially active source windows, before the final stores. No
nonalias premise is needed: source reads finish before the epilogue writes. -/
structure Access (p : Parameters) (initial : State α) (bounds : RegionName → Nat) : Prop where
  seed : ∀ i : Row p,active p initial i → row p initial i<bounds p.seeds
  offset : ∀ i : Row p,active p initial i → row p initial i<bounds p.offsets
  score : ∀ i : Row p,active p initial i → ∀ j,j<p.tokens →
    row p initial i*p.strideScoreBatch+j*p.strideScoreToken<bounds p.scores
  label : ∀ i : Row p,active p initial i → ∀ j,j<max p.tokens 1 →
    row p initial i*p.strideIndexBatch+j*p.strideIndexToken<bounds p.indices
  output : ∀ i : Row p,active p initial i → row p initial i<bounds p.output

theorem head_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (bounds : RegionName → Nat) (access : Access p initial bounds) :
    Safety.traceSafeList (backend rest P) bounds (headCode p) initial := by
  simp [headCode,prologue,source,original,Source.original,Source.original16,Source.originalBfloat16,Source.original64,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,
    numeric,natLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,Option.bind,
    Random.i32ArangeSize]
  intro i active
  exact access.seed (i,()) active

theorem offset_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (bounds : RegionName → Nat) (access : Access p initial bounds) :
    Safety.traceSafeList (backend rest P) bounds (offsetCode p) (headExecution rest P p initial).val := by
  simp [offsetCode,offsetExpr,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,
    headExecution_off,headExecution_mask,headExecution_mem,headExecution_undef,
    numeric,natLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,Option.bind]
  intro i tail active
  cases tail
  exact access.offset (i,()) active

theorem last_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (seeds : SeedDomain p initial)
    (bounds : RegionName → Nat) (access : Access p initial bounds) :
    Safety.traceSafeList (backend rest P) bounds (lastCode p) (offsetExecution rest P p initial).val := by
  have hs (a : Fin p.block) (b : TileIndex []) :
      Random.IntegerType.i64.inRange (seedValue p initial (a,b))=Bool.true := seeds (a,b)
  simp [lastCode,prologue,source,original,Source.original,Source.original16,Source.originalBfloat16,Source.original64,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,
    offsetExecution_off,offsetExecution_mask,offsetExecution_mem,offsetExecution_undef,
    offsetExecution_seed,offsetExecution_offset,
    hs,narrowed32_bmod_inRange,offsetValue,numeric,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,Option.bind,Random.i32ArangeSize]
  intro i tail active
  cases tail
  simpa using access.label (i,()) active 0 (by omega)

theorem prelude_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (seeds : SeedDomain p initial)
    (bounds : RegionName → Nat) (access : Access p initial bounds) :
    Safety.traceSafeList (backend rest P) bounds (prologue .fp32 p) initial := by
  rw [prologue_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨head_safe rest P p initial bounds access,?_⟩
  intro middle ran
  have same : middle=(headExecution rest P p initial).val := by
    simpa only [(headExecution rest P p initial).property,Option.some.injEq] using ran.symm
  subst middle
  rw [Safety.traceSafeList_append]
  refine ⟨offset_safe rest P p initial bounds access,?_⟩
  intro middle ran
  have same : middle=(offsetExecution rest P p initial).val := by
    simpa only [(offsetExecution rest P p initial).property,Option.some.injEq] using ran.symm
  subst middle
  exact last_safe rest P p initial seeds bounds access


theorem body_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (b : Nat) (st : Accumulator α p)
    (h : Invariant P p initial st current) (bounds : RegionName → Nat)
    (access : Access p initial bounds) :
    Safety.traceSafeList (backend rest P) bounds (body .fp32 p)
      (current.setReg "b_idx" .nat [] (fun _=>b)) := by
  have hn : 0<p.blockTokens := p.tokensDomain.1
  simp [body,source,original,Source.original,Source.original16,Source.originalBfloat16,Source.original64,ComputeKernel.surfaceBody,Random.i32ArangeSize,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,
    h.memory,h.off,h.n_off,h.mask,h.samp,h.acc,h.output,
    numeric,natLt,numericLt,numericLe,ofFloat,toFloat,ComputeDType.eraseDType,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,Option.bind,
    TileShape.axisDim,hn,bodyResult,scores,padding,Storage.dtype,readScore,
    scoreOffset,scanned,cumulative,advance,valid,found,selected,selectedIndex,min_def_lt,ite_mul]
  constructor
  · intro i j active token
    exact access.score (i,()) active (b+j.val) token
  · intro i found active
    rw [←ite_mul]
    exact access.label (i,()) active _ (clamped_lt _ _)

theorem epilogue_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (st : Accumulator α p)
    (h : Invariant P p initial st current) (bounds : RegionName → Nat)
    (access : Access p initial bounds) :
    Safety.traceSafeList (backend rest P) bounds (epilogue .fp32 p) current := by
  simp [epilogue,source,original,Source.original,Source.original16,Source.originalBfloat16,Source.original64,ComputeKernel.surfaceBody,Random.i32ArangeSize,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,FP.Structural.evalOp,store,
    h.off,h.mask,h.output,writeBack,Region.cast,Option.bind]
  intro i tail active
  cases tail
  exact access.output (i,()) active


theorem loop_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (k : Nat) (hk : k≤count p)
    (h : Invariant P p initial (rounds P .fp32 p initial k) current)
    (bounds : RegionName → Nat) (access : Access p initial bounds) :
    Safety.rangeSafe (backend rest P) bounds "b_idx" (k*p.blockTokens)
      p.tokens p.blockTokens (body .fp32 p) current := by
  have aux : ∀ fuel k (current : State α),count p-k=fuel → k≤count p →
      Invariant P p initial (rounds P .fp32 p initial k) current →
      Safety.rangeSafe (backend rest P) bounds "b_idx" (k*p.blockTokens)
        p.tokens p.blockTokens (body .fp32 p) current := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro k current he hk h
      rw [Safety.rangeSafe_unfold,if_neg (by have := p.tokensDomain.1; omega)]
      by_cases hb : k<count p
      · rw [if_pos ((before_count p k).mp hb)]
        refine ⟨body_safe rest P p initial current (k*p.blockTokens) _ h bounds access,?_⟩
        rw [bodyExecution rest P .fp32 p initial current (k*p.blockTokens) _ h]
        have hi := bodyInvariant P .fp32 p initial current (k*p.blockTokens) _ h
        have next := ih (count p-(k+1)) (by omega) (k+1) _ rfl (by omega) hi
        simpa only [Nat.add_mul,Nat.one_mul] using next
      · rw [if_neg (by simpa only [←before_count] using hb)]
        trivial
  exact aux _ k current rfl hk h

theorem whole (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (seeds : SeedDomain p initial)
    (bounds : RegionName → Nat) (access : Access p initial bounds) :
    TraceSafe (backend rest P) bounds (source .fp32 p) initial := by
  change Safety.traceSafeList (backend rest P) bounds (source .fp32 p).surfaceBody initial
  rw [source_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe rest P p initial seeds bounds access,?_⟩
  intro start ran
  have same : start=(preludeExecution rest P .fp32 p initial seeds).val := by
    simpa only [(preludeExecution rest P .fp32 p initial seeds).property,Option.some.injEq] using ran.symm
  subst start
  simp only [List.singleton_append,Safety.traceSafeList_cons]
  have inv := preludeInvariant rest P .fp32 p initial seeds
  refine ⟨?_,?_⟩
  · rw [Safety.stepSafe_unfold]
    change Safety.rangeSafe (backend rest P) bounds "b_idx" 0 p.tokens p.blockTokens (body .fp32 p) _
    simpa only [Nat.zero_mul] using loop_safe rest P p initial _ 0 (by omega) inv bounds access
  · intro middle executed
    obtain ⟨finish,run,finalInv⟩ := loopExecution rest P .fp32 p initial _ 0 (by omega) inv
    have stepEq : FP.Structural.step (backend rest P)
        (.forRange "b_idx" 0 p.tokens p.blockTokens (body .fp32 p))
        (preludeExecution rest P .fp32 p initial seeds).val=some finish := by
      simpa only [FP.Structural.step,Nat.zero_mul] using run
    have same : middle=finish := by simpa only [stepEq,Option.some.injEq] using executed.symm
    subst middle
    exact epilogue_safe rest P p initial finish _ finalInv bounds access

#axiomsClean loop_safe
#axiomsClean whole
#axiomsClean body_safe
#axiomsClean epilogue_safe
#axiomsClean head_safe
#axiomsClean offset_safe
#axiomsClean last_safe
#axiomsClean prelude_safe
end
end VeriTile.Bench.Repairs.MultinomialSampling.SourceSafety
