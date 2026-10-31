import bench.tritonbench_g.triton_matmul.Proofs.SourceBodyInvariant
import bench.tritonbench_g.triton_matmul.Proofs.SourceRange
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current)
    (ha : HasAcc p (fun _=>M.literal (some .fp32) .real 0) current) :
    ∃ t,rangeInt (backend M D) "k" 0 (p.numKBlocks : Int) 1 (iteration p st) current=some t ∧
      Context p initial t ∧ HasAcc p (accumulation M D p st initial) t := by
  let I:=fun (_ : Nat) (a : CIndex p→α) (s : State α)=>Context p initial s ∧ HasAcc p a s
  exact nat_range_values (backend M D) "k" (iteration p st) (update M D p st initial)
    I (fun a s=>Context p initial s ∧ HasAcc p a s) 0 p.numKBlocks 1 _ current
    (by
      intro k a s h hk
      let t:=bodyExecution M D p st initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p st initial s k a h.1 h.2⟩)
    (by intro n a s h;exact h) ⟨ctx,ha⟩

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulation M D p st initial) current) :
    {t : State α // run (backend M D) (epilogue p st) current=some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_,?_⟩
     swap
     simp [epilogue,typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,dtype,
       ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
       ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
       evalOffsets,store,numeric,bop,natLt,numericLt,numericLe,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.metadata.2.2.2.2.2.2.2.1,ctx.metadata.2.2.2.2.2.2.2.2,
       ctx.memory,ctx.am,ctx.bn,ctx.mm,ctx.nn,ha,backend]
     rfl)

def sourceOutputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (i : CIndex p) : α :=
  outputValue M D p st s i

def cSafeOffset {α : Type} (p : Parameters) (initial : State α) (i : CIndex p) :=
  p.stride_cm*(if row p initial i.1<p.M then row p initial i.1 else 0)+
    p.stride_cn*(if col p initial i.2.1<p.N then col p initial i.2.1 else 0)

def writtenSafe {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i=>if active p initial i then
    t.write p.C (cSafeOffset p initial i) (stored st (sourceOutputValue M D p st initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulation M D p st initial) current) :
    (epilogueExecution M D p st initial current ctx ha).val.mem=(writtenSafe M D p st initial).mem := by
  cases st <;> simp [epilogueExecution,writtenSafe,masked_scatter_memory,ctx.memory,
    active,cSafeOffset,aRow,bCol,row,col,sourceOutputValue,outputValue,stored,dtype,backend,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  all_goals apply congrArg (fun f=>List.foldl f initial.mem (TileShape.allIndices [BM p,BN p]))
  all_goals funext mem i
  all_goals by_cases hr:pidM p initial*BM p+i.1.val<p.M
  all_goals by_cases hc:pidN p initial*BN p+i.2.1.val<p.N
  all_goals simp only [hr,hc,↓reduceIte,and_true,true_and,false_and,and_false,
    ←Int.natCast_add,Int.toNat_natCast]


#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
