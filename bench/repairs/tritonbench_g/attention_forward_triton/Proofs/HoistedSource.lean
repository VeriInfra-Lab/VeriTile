import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.Output

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (p : Parameters) : ComputeKernel :=
  TypedSource.hoistedHeadMask p.Q p.K p.V p.QS p.KS p.Out
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.batches p.heads p.length 96 128 64 128 96 1

def parts (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernel p).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

structure Context (M : Algebra α) (p : Parameters) (initial : State α)
    (row : Nat) (a : Acc α) (current : State α) extends Source.Context M p initial row a current where
  headMask : current.regs .bool [128] "head_mask"=some (fun i=>decide (i.1.val<96))

def entry (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M dot)) (parts p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.hoistedHeadMask,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
  rfl

@[simp] theorem entry_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    (entry M dot p s).val.mem=s.mem := by simp [entry]

theorem entry_context (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    Context M p s 0 (initialAcc M) (entry M dot p s).val := by
  refine ⟨?_,?_⟩
  · constructor
    all_goals simp [entry,query,queryScale,base,zero,initialAcc,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
    all_goals funext i
    all_goals simp [query,queryScale,base,zero,initialAcc,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
    all_goals norm_num
    all_goals rfl

  · simp only [entry]
    rfl

#axiomsClean entry
#axiomsClean entry_context
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedSource
