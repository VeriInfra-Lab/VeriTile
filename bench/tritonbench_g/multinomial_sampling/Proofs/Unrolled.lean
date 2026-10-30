import bench.tritonbench_g.multinomial_sampling.Proofs.IOComplete
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MultinomialSampling.Unrolled
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (p : Parameters) : ComputeKernel :=
  Source.unrolled p.scores p.seeds p.offsets p.indices p.output
    p.strideScoreBatch p.strideScoreToken p.strideIndexBatch p.strideIndexToken
    p.batches p.tokens p.block p.blockTokens p.blockDomain p.tokensDomain

def chunks (p : Parameters) := (p.tokens+p.blockTokens-1)/p.blockTokens

def body (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel p).surfaceBody
  match code[11]? with
  | some (ComputeStmt.forLoop _ _ b)=>b
  | _=>[]

def scale (p : Parameters) : ComputeStmt := (body p).headD (.effectMarker "unreachable")

theorem guard (p : Parameters) (n : Nat) : n<chunks p ↔ n*p.blockTokens<p.tokens := by
  unfold chunks
  rw [Nat.lt_div_iff_mul_lt p.tokensDomain.1]
  have h : p.tokens+p.blockTokens-1-(p.blockTokens-1)=p.tokens := by
    have := p.tokensDomain.1
    omega
  rw [h]

theorem overwrite (s : State α) (n : RegName) (d : TileDType) (sh : TileShape)
    (u v : Values α d sh) : (s.setReg n d sh u).setReg n d sh v=s.setReg n d sh v := by
  simp only [State.setReg]
  congr 1
  funext d' sh' n'
  split <;> simp_all

theorem body_split (p : Parameters) : body p=scale p::SourceValue.body .fp32 p := rfl

theorem scale_run (M : Algebra α) (p : Parameters) (s : State α) (n : Nat) :
    step M (scale p) (s.setReg "b_idx" .nat [] (fun _=>n))=
      some (s.setReg "b_idx" .nat [] (fun _=>n*p.blockTokens)) := by
  simp [scale,body,kernel,Source.unrolled,ComputeKernel.surfaceBody,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,overwrite]
  rfl

theorem body_run (M : Algebra α) (p : Parameters) (s : State α) (n : Nat) :
    run M (body p) (s.setReg "b_idx" .nat [] (fun _=>n))=
      run M (SourceValue.body .fp32 p) (s.setReg "b_idx" .nat [] (fun _=>n*p.blockTokens)) := by
  rw [body_split,run,scale_run]
  rfl

theorem scale_safe (M : Algebra α) (p : Parameters) (cap : RegionName → Nat) (s : State α) :
    Safety.stepSafe M cap (scale p) s := by
  simp [scale,body,kernel,Source.unrolled,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

theorem body_safe (M : Algebra α) (p : Parameters) (cap : RegionName → Nat) (s : State α) (n : Nat) :
    Safety.traceSafeList M cap (body p) (s.setReg "b_idx" .nat [] (fun _=>n)) ↔
      Safety.traceSafeList M cap (SourceValue.body .fp32 p)
        (s.setReg "b_idx" .nat [] (fun _=>n*p.blockTokens)) := by
  rw [body_split,Safety.traceSafeList_cons]
  simp only [scale_safe,scale_run,Option.some.injEq,true_and]
  simp

theorem loops (M : Algebra α) (p : Parameters) (n : Nat) (s : State α) :
    FP.Structural.loop M "b_idx" n (chunks p) (body p) s=
      FP.Structural.range M "b_idx" (n*p.blockTokens) p.tokens p.blockTokens (SourceValue.body .fp32 p) s ∧
    ∀ cap,Safety.loopSafe M cap "b_idx" n (chunks p) (body p) s ↔
      Safety.rangeSafe M cap "b_idx" (n*p.blockTokens) p.tokens p.blockTokens (SourceValue.body .fp32 p) s := by
  have positive:=p.tokensDomain.1
  have nonzero : p.blockTokens≠0 := by omega
  by_cases active : n<chunks p
  · have live: n*p.blockTokens<p.tokens := (guard p n).mp active
    have nextIndex : n*p.blockTokens+p.blockTokens=(n+1)*p.blockTokens := by simp [Nat.add_mul]
    constructor
    · rw [FP.Structural.loop,FP.Structural.range]
      simp only [active,live,nonzero,if_true,if_false,body_run]
      cases he : run M (SourceValue.body .fp32 p) (s.setReg "b_idx" .nat [] (fun _=>n*p.blockTokens)) with
      | none=>rfl
      | some t=>
        rw [nextIndex]
        exact (loops M p (n+1) t).1
    · intro cap
      rw [Safety.loopSafe,Safety.rangeSafe]
      simp only [active,live,nonzero,if_true,if_false,body_run]
      cases he : run M (SourceValue.body .fp32 p) (s.setReg "b_idx" .nat [] (fun _=>n*p.blockTokens)) with
      | none=>exact and_congr (body_safe M p cap s n) Iff.rfl
      | some t=>
        rw [nextIndex]
        exact and_congr (body_safe M p cap s n) ((loops M p (n+1) t).2 cap)
  · have dead : ¬n*p.blockTokens<p.tokens := mt (guard p n).mpr active
    constructor
    · rw [FP.Structural.loop,FP.Structural.range]
      simp only [active,dead,nonzero,if_false]
    · intro cap
      rw [Safety.loopSafe,Safety.rangeSafe]
      simp only [active,dead,nonzero,if_false]
termination_by chunks p-n
decreasing_by all_goals omega

theorem source_shape (p : Parameters) :
    (kernel p).surfaceBody=prologue .fp32 p ++
      [.forLoop "b_idx" (chunks p) (body p)] ++ epilogue .fp32 p := rfl

theorem statement_run (M : Algebra α) (p : Parameters) (s : State α) :
    step M (.forRange "b_idx" 0 p.tokens p.blockTokens (SourceValue.body .fp32 p)) s=
      step M (.forLoop "b_idx" (chunks p) (body p)) s := by
  exact (by simpa only [step,Nat.zero_mul] using (loops M p 0 s).1.symm)

theorem statement_safe (M : Algebra α) (p : Parameters) (cap : RegionName → Nat) (s : State α) :
    Safety.stepSafe M cap (.forRange "b_idx" 0 p.tokens p.blockTokens (SourceValue.body .fp32 p)) s ↔
      Safety.stepSafe M cap (.forLoop "b_idx" (chunks p) (body p)) s := by
  simpa only [Safety.stepSafe_unfold,Nat.zero_mul] using ((loops M p 0 s).2 cap).symm

theorem execution_equal (M : Algebra α) (p : Parameters) (s : State α) :
    FP.Structural.exec M (MultinomialSampling.kernel p) s=FP.Structural.exec M (kernel p) s := by
  have left : FP.Structural.exec M (MultinomialSampling.kernel p) s=
      run M (source .fp32 p).surfaceBody s := rfl
  have right : FP.Structural.exec M (kernel p) s=run M (kernel p).surfaceBody s := by
    cases kernel p; rfl
  rw [left,right,source_decomposition,source_shape,List.append_assoc,List.append_assoc]
  simp only [run_append,List.singleton_append,run,statement_run]

theorem safety_equal (M : Algebra α) (p : Parameters) (cap : RegionName → Nat) (s : State α) :
    TraceSafe M cap (MultinomialSampling.kernel p) s ↔ TraceSafe M cap (kernel p) s := by
  change Safety.traceSafeList M cap (source .fp32 p).surfaceBody s ↔ _
  rw [show TraceSafe M cap (kernel p) s=Safety.traceSafeList M cap (kernel p).surfaceBody s from rfl]
  rw [source_decomposition,source_shape,List.append_assoc,List.append_assoc]
  simp only [Safety.traceSafeList_append,List.singleton_append,Safety.traceSafeList_cons,
    statement_run,statement_safe]

#axiomsClean execution_equal
#axiomsClean safety_equal
end
end VeriTile.Bench.Repairs.MultinomialSampling.Unrolled
