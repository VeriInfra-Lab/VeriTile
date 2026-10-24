import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Middle
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prefixAcc (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) : Acc α :=
  foldRange (update M dot p s) 0 (s.pids 0*128) 64 (initialAcc M)

def accumulated (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) : Acc α :=
  foldRange (update M dot p s) (s.pids 0*128) (stop p s) 64 (prefixAcc M dot p s)

theorem loopAt (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (cur finish : Nat) (a : Acc α) (ctx : Context M p initial cur a current) :
    ∃ t, range (engine (withDot M dot)) "start_n" cur finish 64 (parts p).body current=some t ∧
      ∃ row, Context M p initial row (foldRange (update M dot p initial) cur finish 64 a) t := by
  exact range_values (engine (withDot M dot)) "start_n" (parts p).body (update M dot p initial)
    (fun row a s => Context M p initial row a s)
    (fun a s => ∃ row, Context M p initial row a s) cur finish 64 _ current
    (by
      intro row a s h _
      let t:=bodyExec M dot p initial s row a h
      exact ⟨t.val,t.property,body_context M dot p initial s row a h⟩)
    (by intro row a s h; exact ⟨row,h⟩) ctx

def firstStatement (p : Parameters) : ComputeStmt :=
  .forRangeDyn "start_n" (parts p).start (parts p).stop (parts p).stride (parts p).body
def secondStatement (p : Parameters) : ComputeStmt :=
  .forRangeDyn "start_n" (diagonal p).start (diagonal p).stop (diagonal p).stride (diagonal p).body

theorem bodies_equal (p : Parameters) : (diagonal p).body=(parts p).body := by rfl

theorem source_body (p : Parameters) : (kernel p).surfaceBody=
    (parts p).before++[firstStatement p]++middle p++[secondStatement p]++(diagonal p).after := by rfl

theorem first_step (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α) (ctx : Context M p initial row a current) :
    step (engine (withDot M dot)) (firstStatement p) current=
      range (engine (withDot M dot)) "start_n" 0 (initial.pids 0*128) 64 (parts p).body current := by
  simp [firstStatement,step,parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    evalOp_unfold,ctx.firstLo,ctx.firstHi]

theorem second_step (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α) (ctx : Context M p initial row a current) :
    step (engine (withDot M dot)) (secondStatement p) (middleExec M dot p initial current row a ctx).val=
      range (engine (withDot M dot)) "start_n" (initial.pids 0*128) (stop p initial) 64
        (parts p).body (middleExec M dot p initial current row a ctx).val := by
  have hm : (if (initial.pids 0+1)*128<p.length then (initial.pids 0+1)*128 else p.length)=
      min ((initial.pids 0+1)*128) p.length := by split <;> omega
  simp [hm,secondStatement,step,diagonal,parts,kernel,TypedSource.repaired,stop,middleExec,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    evalOp_unfold,State.setReg,FP.Structural.bop,numeric,natLe,ofFloat,toFloat]

theorem foldRange_append {β : Type} (f : Nat → β → β) (cur split finish : Nat) (a : β)
    (hc : cur≤split) (ha : (split-cur)%64=0) (hf : split≤finish) :
    foldRange f cur finish 64 a=foldRange f split finish 64 (foldRange f cur split 64 a) := by
  by_cases h : cur<split
  · have hs : cur+64≤split := by omega
    have hfinish : cur<finish := by omega
    have ih:=foldRange_append f (cur+64) split finish (f cur a) hs (by omega) hf
    have hp : foldRange f cur split 64 a=foldRange f (cur+64) split 64 (f cur a) := by
      rw [foldRange];simp [h]
    have ht : foldRange f cur finish 64 a=foldRange f (cur+64) finish 64 (f cur a) := by
      rw [foldRange];simp [hfinish]
    rw [ht,hp]
    exact ih
  · have he : cur=split := by omega
    subst cur
    have hz : foldRange f split split 64 a=a := by rw [foldRange];simp
    rw [hz]
termination_by split-cur
decreasing_by omega

#axiomsClean foldRange_append
#specNonCircular accumulated avoiding [run,FP.Structural.exec,bodyExec,loopAt]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source
