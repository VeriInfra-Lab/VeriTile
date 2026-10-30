import bench.tritonbench_g.attention_kernel.Proofs.Loop
import bench.tritonbench_g.attention_kernel.Proofs.Scatter
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Scatter (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false

def active (p : Parameters) (s : State α) (i : TileIndex [p.bm,p.dim]) : Prop :=
  s.pids 0*p.bm+i.1.val<p.length
instance (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _=>inferInstanceAs (Decidable (_<_))
def address (p : Parameters) (s : State α) (i : TileIndex [p.bm,p.dim]) : Nat :=
  base p s p.o+(s.pids 0*p.bm+i.1.val)*p.o.row+i.2.1.val*p.o.dim

def outputValue (M : Algebra α) (p : Parameters) (a : Acc p α) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .real p.storage.float
    (M.binary (some .fp32) .real .div (a.value i) (a.normalizer i.1))

theorem output_mask (p : Parameters) (s : State α) (i : TileIndex [p.bm,p.dim]) :
    (optr p s).inBounds (TileShape.indexToList _ i) [0]=decide (active p s i) := by
  change (decide (s.pids 0*p.bm+i.1.val<p.length) && Bool.true)=_
  simp [active]

theorem check0 (r : RegionName) (base N D BM rs ds start : Nat) (i : TileIndex [BM,D]) :
    (BlockPtr.mk r base [N,D] [BM,D] [rs,ds] [start,0]).inBounds
      (TileShape.indexToList _ i) [0]=decide (start+i.1.val<N) := by
  change (decide (start+i.1.val<N) && Bool.true)=_
  simp

def postExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).after current=some t} := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,partsAs,kernel,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,ctx.memory,ctx.startM,ctx.outputOffset,
      ctx.normalizer,ctx.value,FP.Structural.store]
    rfl

def written (M : Algebra α) (p : Parameters) (s : State α) (a : Acc p α) : State α :=
  (TileShape.allIndices [p.bm,p.dim]).foldl (fun t i=>if active p s i then
    t.write p.Out (address p s i) (.mk p.storage.tile (ofFloat p.storage.float (outputValue M p a i))) else t) s

theorem post_memory (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial row a current) :
    (postExec M dot p initial current row a ctx).val.mem=(written M p initial a).mem := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  rcases Out with ⟨outName⟩
  cases storage
  all_goals simp [postExec,written,masked_scatter_memory,ctx.memory,active,address,outputValue,
    TypedSource.Storage.float,TypedSource.Storage.tile,FloatDType.toTileDType,ofFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    check0,optr,base,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals congr 1
  all_goals funext mem i
  all_goals
    have he : initial.pids 1/heads*o.batch+(initial.pids 1%heads*o.head+(i.1.val+initial.pids 0*bm)*o.row)=
      (i.1.val+initial.pids 0*bm)*o.row+(initial.pids 1/heads*o.batch+initial.pids 1%heads*o.head) := by omega
    rw [he]

/-- Every source statement executes from current memory. No whole-output,
reduction, dot-value or floating-equivalence premise appears here. -/
theorem source_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (hbn : 0<p.bn) :
    ∃ t,FP.Structural.exec (engine (withDot M dot)) (kernel p) s=some t ∧
      t.mem=(written M p s (accumulated M dot p s)).mem := by
  let start:=entry M dot p s
  obtain ⟨mid,hm,row,hc⟩:=loopExec M dot p s start.val hbn (entry_context M dot p s)
  let finish:=postExec M dot p s mid row (accumulated M dot p s) hc
  refine ⟨finish.val,?_,post_memory M dot p s mid row _ hc⟩
  have hl : step (engine (withDot M dot)) (sourceLoop p) start.val=some mid := by
    simpa [sourceLoop,step,evalOp_unfold,(entry_bounds M dot p s).1,(entry_bounds M dot p s).2,start] using hm
  change run (engine (withDot M dot)) (kernel p).surfaceBody s=some finish.val
  have hs : run (engine (withDot M dot)) (parts p).before s=some start.val := start.property
  rw [source_body,List.append_assoc,run_append,hs]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]

theorem readback (M : Algebra α) (p : Parameters) (s : State α) (a : Acc p α)
    (i : TileIndex [p.bm,p.dim]) (hi : active p s i)
    (unique : ∀ j k,active p s j→active p s k→address p s j=address p s k→j=k) :
    (written M p s a).mem p.Out (address p s i)=
      .mk p.storage.tile (ofFloat p.storage.float (outputValue M p a i)) :=
  masked_readback_active _ _ _ _ _ i hi unique

theorem frame (M : Algebra α) (p : Parameters) (s : State α) (a : Acc p α)
    (r : RegionName) (off : Nat) (h : r≠p.Out ∨ ∀ i,active p s i→ off≠address p s i) :
    (written M p s a).mem r off=s.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right;intro i _ hi;exact ho i hi

#specNonCircular outputValue avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
