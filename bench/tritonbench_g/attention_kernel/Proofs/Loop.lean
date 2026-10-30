import bench.tritonbench_g.attention_kernel.Proofs.Values
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

def accumulated (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) : Acc p α :=
  foldRange (update M dot p s) 0 p.length p.bn (initialAcc M p)

theorem loopExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (hbn : 0<p.bn) (ctx : Context M p initial 0 (initialAcc M p) current) :
    ∃ t, range (engine (withDot M dot)) "start_n" 0 p.length p.bn (parts p).body current=some t ∧
      ∃ row,Context M p initial row (accumulated M dot p initial) t := by
  exact range_values (engine (withDot M dot)) "start_n" (parts p).body (update M dot p initial)
    (fun row a s=>Context M p initial row a s)
    (fun a s=>∃ row,Context M p initial row a s) 0 p.length p.bn _ current
    (by
      intro row a s h _
      let t:=bodyExec M dot p initial s row a hbn h
      exact ⟨t.val,t.property,body_context M dot p initial s row a hbn h⟩)
    (by intro row a s h;exact ⟨row,h⟩) ctx

def sourceLoop (p : Parameters) : ComputeStmt :=
  .forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat p.bn) (parts p).body

theorem source_body (p : Parameters) :
    (kernel p).surfaceBody=(parts p).before++[sourceLoop p]++(parts p).after := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage <;> rfl

theorem entry_bounds (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    (entry M dot p s).val.regs .nat [] "lo"=some (fun _=>0) ∧
    (entry M dot p s).val.regs .nat [] "hi"=some (fun _=>p.length) := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage <;> simp [entry,entryAs]

#specNonCircular accumulated avoiding [run,FP.Structural.exec,bodyExec,loopExec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
