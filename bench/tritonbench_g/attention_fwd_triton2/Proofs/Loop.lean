import bench.tritonbench_g.attention_fwd_triton2.Proofs.Values
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

def accumulated (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) : Acc α :=
  foldRange (update M dot p s) 0 p.length 64 (initialAcc M)

theorem loopExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (ctx : Context M p initial 0 (initialAcc M) current) :
    ∃ t, range (engine (withDot M dot)) "start_n" 0 p.length 64 (parts p).body current=some t ∧
      ∃ row, Context M p initial row (accumulated M dot p initial) t := by
  exact range_values (engine (withDot M dot)) "start_n" (parts p).body (update M dot p initial)
    (fun row a s => Context M p initial row a s)
    (fun a s => ∃ row, Context M p initial row a s) 0 p.length 64 _ current
    (by
      intro row a s h _
      let t:=bodyExec M dot p initial s row a h
      exact ⟨t.val,t.property,body_context M dot p initial s row a h⟩)
    (by intro row a s h; exact ⟨row,h⟩) ctx

theorem source_body (p : Parameters) : (kernel p).surfaceBody=
    (parts p).before++[.forRange "start_n" 0 p.length 64 (parts p).body]++(parts p).after := by
  rfl

#specNonCircular accumulated avoiding [run,FP.Structural.exec,bodyExec,loopExec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
