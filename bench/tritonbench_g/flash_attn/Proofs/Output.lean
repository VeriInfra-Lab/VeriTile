import bench.tritonbench_g.flash_attn.Proofs.Loop
import bench.tritonbench_g.attention_kernel.Proofs.Scatter

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionKernel.Whole001.Scatter
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false

def active (p : Parameters) (s : State α) (i : TileIndex [128,p.DIM]) : Prop :=
  s.pids 0*128+i.1.val<p.SEQLEN
instance (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _=>inferInstanceAs (Decidable (_<_))
def address (p : Parameters) (s : State α) (i : TileIndex [128,p.DIM]) : Nat :=
  base p s p.stride_o_bs p.stride_o_head+
    (s.pids 0*128+i.1.val)*p.stride_o_seqlen+i.2.1.val*p.stride_o_dim

def lActive (p : Parameters) (s : State α) (i : TileIndex [128]) : Prop :=
  s.pids 0*128+i.1.val<p.SEQLEN
instance (p : Parameters) (s : State α) : DecidablePred (lActive p s) :=
  fun _=>inferInstanceAs (Decidable (_<_))
def lAddress (p : Parameters) (s : State α) (i : TileIndex [128]) : Nat :=
  s.pids 1*p.SEQLEN+(s.pids 0*128+i.1.val)

def outputValue (M : Algebra α) (p : Parameters) (a : Acc p α) (i : TileIndex [128,p.DIM]) : α :=
  M.cast (some .fp32) .fp16 .real (M.cast (some .fp32) .real .fp16
    (M.binary (some .fp32) .real .div (a.output i) (a.normalizer i.1)))
def logValue (M : Algebra α) (p : Parameters) (a : Acc p α) (i : TileIndex [128]) : α :=
  M.binary (some .fp32) .real .add (a.maximum i.1) (M.unary (some .fp32) .log2 (a.normalizer i.1))

theorem check0 (r : RegionName) (base N D rs ds start : Nat) (i : TileIndex [128,D]) :
    (BlockPtr.mk r base [N,D] [128,D] [rs,ds] [start,0]).inBounds
      (TileShape.indexToList _ i) [0]=decide (start+i.1.val<N) := by
  change (decide (start+i.1.val<N) && Bool.true)=_
  simp

def postExec (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial scale row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).after current=some t} := by
  rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
  cases causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 1000000) [parts,kernel,declared,ComputeKernel.surfaceBody,
      Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,ctx.memory,ctx.startM,ctx.head,ctx.rows,
      ctx.outputOffset,ctx.maximum,ctx.normalizer,ctx.output,FP.Structural.store]
    rfl

def written (M : Algebra α) (p : Parameters) (s : State α) (a : Acc p α) : State α :=
  let scratch:=(TileShape.allIndices [128]).foldl (fun t i=>if lActive p s i then
    t.write "L" (lAddress p s i) (.mk .real (logValue M p a i)) else t) s
  (TileShape.allIndices [128,p.DIM]).foldl (fun t i=>if active p s i then
    t.write "O" (address p s i) (.mk .real (outputValue M p a i)) else t) scratch

theorem post_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial scale row a current) :
    (postExec M dot p scale initial current row a ctx).val.mem=(written M p initial a).mem := by
  rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
  cases causal
  all_goals simp [postExec,written,masked_scatter_memory,ctx.memory,active,address,lActive,lAddress,
    outputValue,logValue,FloatDType.toTileDType,ofFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    check0,optr,base,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals congr 1
  all_goals funext mem i
  all_goals
    have he : initial.pids 1/heads*obs+(initial.pids 1%heads*oh+(i.1.val+initial.pids 0*128)*orr)=
      (i.1.val+initial.pids 0*128)*orr+(initial.pids 1/heads*obs+initial.pids 1%heads*oh) := by omega
    rw [he]

/-- Complete execution of the selected quoted source, including the original
loop and both stores, from arbitrary initial payloads and incoming scale. -/
theorem source_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α) :
    ∃ t,FP.Structural.exec (engine (withDot M dot)) (kernel p) (bindScale scale s)=some t ∧
      t.mem=(written M p s (accumulated M dot p s scale)).mem := by
  let start:=entry M dot p scale s
  obtain ⟨mid,hm,row,hc⟩:=loopExec M dot p scale s start.val (entry_context M dot p scale s)
  let finish:=postExec M dot p scale s mid row (accumulated M dot p s scale) hc
  refine ⟨finish.val,?_,post_memory M dot p scale s mid row _ hc⟩
  have hl : step (engine (withDot M dot))
      (.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat 64) (parts p).body)
      start.val=some mid := by
    simpa [step,evalOp_unfold,(entry_bounds M dot p scale s).1,(entry_bounds M dot p scale s).2,start] using hm
  change run (engine (withDot M dot)) (kernel p).surfaceBody (bindScale scale s)=some finish.val
  have hs : run (engine (withDot M dot)) (parts p).before (bindScale scale s)=some start.val := start.property
  rw [source_parts,List.append_assoc,run_append,hs]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]

#axiomsClean source_memory
#specNonCircular outputValue avoiding [run,FP.Structural.exec,postExec]
#specNonCircular logValue avoiding [run,FP.Structural.exec,postExec]
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
