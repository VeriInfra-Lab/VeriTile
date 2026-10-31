import bench.tritonbench_g.triton_matmul.Proofs.SourceInitialValues
import bench.tritonbench_g.triton_matmul.Proofs.SourceCounts
import bench.tritonbench_g.triton_matmul.Proofs.SourceControl
import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def coordinates (p : Parameters) (st : Storage) := (prelude p st).drop 22

theorem signed_mask (limit base lane : Nat) :
    decide ((lane : Int)<(limit : Int)-(base : Int))=decide (base+lane<limit) := by
  have h : (lane : Int)<(limit : Int)-(base : Int) ↔ base+lane<limit := by omega
  simp only [h]

theorem signed_clamp (limit base lane : Nat) :
    (if (lane : Int)<(limit : Int)-(base : Int) then (base : Int) else 0)+
      (if (lane : Int)<(limit : Int)-(base : Int) then (lane : Int) else 0)=
      (if base+lane<limit then (base : Int)+(lane : Int) else 0) := by
  by_cases h:base+lane<limit
  · have hi:(lane : Int)<(limit : Int)-(base : Int):=by omega
    simp only [h,hi,↓reduceIte,Int.natCast_add]
  · have hi:¬(lane : Int)<(limit : Int)-(base : Int):=by omega
    simp only [h,hi,↓reduceIte,Int.natCast_zero,add_zero]

structure Context {α : Type} (p : Parameters) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  metadata : Metadata p current
  offk : current.regs .int [BK p] "offs_k"=some (fun i=>(i.1.val : Int))
  am : current.regs .int [BM p] "offs_am"=some (fun i=>(aRow p initial i.1 : Int))
  bn : current.regs .int [BN p] "offs_bn"=some (fun i=>(bCol p initial i.1 : Int))
  mm : current.regs .bool [BM p] "mask_m"=some (fun i=>decide (row p initial i.1<p.M))
  nn : current.regs .bool [BN p] "mask_n"=some (fun i=>decide (col p initial i.1<p.N))
  blocks : current.regs .int [] "blocks_k"=some (fun _=>(ceilCount p.K (BK p) : Int))

-- Algorithmic accumulator predicate, independent of source statement lists.
def HasAcc {α : Type} (p : Parameters) (a : CIndex p→α) (s : State α) : Prop :=
  s.regs .real [BM p,BN p] "accumulator"=some a

def coordinateExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (metadata : Metadata p current)
    (pm : current.regs .int [] "pid_m"=some (fun _=>(pidM p initial : Int)))
    (pn : current.regs .int [] "pid_n"=some (fun _=>(pidN p initial : Int))) :
    {t : State α // run (backend M D) (coordinates p st) current=some t} := by
  rcases metadata with ⟨hm,hn,hk,ham,hak,hbk,hbn,hcm,hcn⟩
  cases st <;>
    (refine ⟨?_,?_⟩
     swap
     simp [coordinates,prelude,typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,dtype,
       ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
       ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
       evalOffsets,store,numeric,bop,natLt,numericLt,numericLe,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       hm,hn,hk,ham,hak,hbk,hbn,hcm,hcn,pm,pn]
     rfl)

theorem coordinateContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (memory : current.mem=initial.mem) (metadata : Metadata p current)
    (pm : current.regs .int [] "pid_m"=some (fun _=>(pidM p initial : Int)))
    (pn : current.regs .int [] "pid_n"=some (fun _=>(pidN p initial : Int))) :
    let t := (coordinateExecution M D p st initial current metadata pm pn).val
    Context p initial t ∧ HasAcc p (fun _=>M.literal (some .fp32) .real 0) t := by
  rcases metadata with ⟨hm,hn,hk,ham,hak,hbk,hbn,hcm,hcn⟩
  cases st <;> constructor
  all_goals try constructor
  all_goals simp [coordinateExecution,Metadata,HasAcc,memory,hm,hn,hk,ham,hak,hbk,hbn,hcm,hcn,
    pm,pn,backend,row,col,aRow,bCol,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals funext i
  all_goals simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try exact SourceControl.wide_ceil _ _
  all_goals try simpa only [Int.natCast_mul,Int.zero_add,Int.add_zero] using signed_mask _ _ _
  all_goals first
    | simpa only [BM,Int.natCast_mul,Int.zero_add,Int.add_zero] using signed_clamp p.M (pidM p initial*BM p) i.1.val
    | simpa only [BN,Int.natCast_mul,Int.zero_add,Int.add_zero] using signed_clamp p.N (pidN p initial*BN p) i.1.val

#axiomsClean coordinateContext

#axiomsClean signed_mask
#axiomsClean signed_clamp
#axiomsClean coordinateExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
