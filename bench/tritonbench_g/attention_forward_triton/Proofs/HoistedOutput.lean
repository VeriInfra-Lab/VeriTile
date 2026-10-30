import bench.tritonbench_g.attention_forward_triton.Proofs.HoistedLoop
import bench.tritonbench_g.attention_forward_triton.Proofs.Scatter
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source
open Scatter (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def postExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).after current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.hoistedHeadMask,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.headMask,ctx.memory,ctx.rows,ctx.outputPointer,ctx.normalizer,ctx.value,FP.Structural.store]
  rfl

theorem post_memory (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    (postExec M dot p initial current row a ctx).val.mem=(written M p initial a).mem := by
  simp [postExec,written,masked_scatter_memory,ctx.memory,active,address,outputValue,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

/-- Entire actual typed source, including initialization, every key block,
normalization, final half cast and masked stores, executes from current memory. -/
theorem source_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M dot)) (kernel p) s=some t ∧
      t.mem=(written M p s (accumulated M dot p s)).mem := by
  let start:=entry M dot p s
  obtain ⟨mid,hm,row,hc⟩:=loopExec M dot p s start.val (entry_context M dot p s)
  let finish:=postExec M dot p s mid row (accumulated M dot p s) hc
  refine ⟨finish.val,?_,post_memory M dot p s mid row _ hc⟩
  have hl : step (engine (withDot M dot)) (.forRange "start_n" 0 p.length 64 (parts p).body)
      start.val=some mid := by simpa only [step] using hm
  change run (engine (withDot M dot)) (kernel p).surfaceBody s=some finish.val
  rw [source_body,List.append_assoc,run_append,start.property]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]


#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedSource
