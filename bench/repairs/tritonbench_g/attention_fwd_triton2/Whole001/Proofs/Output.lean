import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.Loop
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.Scatter
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Scatter (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def active (p : Parameters) (s : State α) (i : TileIndex [128,128]) : Prop :=
  s.pids 0*128+i.1.val<p.length ∧ i.2.1.val<96
instance (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))
def address (p : Parameters) (s : State α) (i : TileIndex [128,128]) : Nat :=
  base p s p.o+(s.pids 0*128+i.1.val)*p.o.row+i.2.1.val*p.o.dim

def outputValue (M : Algebra α) (a : Acc α) (i : TileIndex [128,128]) : α :=
  M.cast (some .fp32) .real .bf16
    (M.binary (some .fp32) .real .div (a.value i) (a.normalizer i.1))

def postExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).after current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.rows,ctx.outputPointer,ctx.normalizer,ctx.value,FP.Structural.store]
  rfl

def written (M : Algebra α) (p : Parameters) (s : State α) (a : Acc α) : State α :=
  (TileShape.allIndices [128,128]).foldl (fun t i =>
    if active p s i then t.write p.Out (address p s i) (.mk .bf16 (outputValue M a i)) else t) s

theorem post_memory (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    (postExec M dot p initial current row a ctx).val.mem=(written M p initial a).mem := by
  simp [postExec,written,masked_scatter_memory,ctx.memory,active,address,outputValue,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

/-- Entire actual typed source, including initialization, every key block,
normalization, final BF16 cast and masked stores, executes from current memory. -/
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

theorem readback (M : Algebra α) (p : Parameters) (s : State α) (a : Acc α)
    (i : TileIndex [128,128]) (hi : active p s i)
    (unique : ∀ j k, active p s j → active p s k → address p s j=address p s k → j=k) :
    (written M p s a).mem p.Out (address p s i)=.mk .bf16 (outputValue M a i) := by
  exact masked_readback_active _ _ _ _ _ i hi unique

theorem frame (M : Algebra α) (p : Parameters) (s : State α) (a : Acc α)
    (r : RegionName) (off : Nat) (h : r≠p.Out ∨ ∀ i, active p s i → off≠address p s i) :
    (written M p s a).mem r off=s.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

#specNonCircular outputValue avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
