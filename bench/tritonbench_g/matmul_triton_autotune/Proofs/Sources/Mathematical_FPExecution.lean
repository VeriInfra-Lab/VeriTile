import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_FPPrelude
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Positive launch tile sizes and the existing grouped-grid bound. These are
proved from the selected launch configuration before applying the Real result. -/
structure Geometry {α : Type} (p : Parameters) (s : State α) : Prop where
  bm : 0 < BM p
  bn : 0 < BN p
  first : firstM p s ≤ blocksM p

theorem cdiv_int (n b : Nat) (hb : 0 < b) :
    ((n : Int) + (b : Int) - 1) / (b : Int) = (((n+b-1)/b : Nat) : Int) := by
  have h : 1 ≤ n+b := by omega
  rw [← Int.natCast_add, ← Int.natCast_one, ← Int.natCast_sub h, Int.natCast_div]

theorem cast_not_neg (x : Nat) : ¬ (x : Int)<0 := by omega

theorem nat_remainder (x b : Nat) :
    (x : Int)-((b*(x/b) : Nat) : Int)=((x%b : Nat) : Int) := by
  simp only [Int.natCast_mul,Int.natCast_div]
  rw [←Int.emod_def,←Int.natCast_mod]

theorem wide_min (x y : Nat) :
    (if (x : Int)<(y : Int) then (x : Int) else (y : Int))=(min x y : Nat) := by
  by_cases h:x<y
  · simp [h,show (x : Int)<(y : Int) by exact_mod_cast h,Nat.min_eq_left (Nat.le_of_lt h)]
  · simp [h,show ¬(x : Int)<(y : Int) by exact_mod_cast h,Nat.min_eq_right (Nat.le_of_not_gt h)]

theorem loopStop_value (p : Parameters) : loopStop p=(p.numKBlocks : Int) := by
  by_cases hz : BK p=0
  · simp [loopStop,Parameters.numKBlocks,hz]
  · have hb : 0<BK p := by omega
    have hn : ¬ (p.K : Int)+((BK p : Int)-1)<0 := by omega
    rw [loopStop,if_neg hn]
    simpa only [←Int.add_sub_assoc,Parameters.numKBlocks] using cdiv_int p.K (BK p) hb

structure Context {α : Type} (p : Parameters) (initial : State α) (_k : Nat) (current : State α) : Prop where
  memory : current.mem=initial.mem
  pm : current.regs .int [] "pid_m"=some (fun _ => (pidM p initial : Int))
  pn : current.regs .int [] "pid_n"=some (fun _ => (pidN p initial : Int))
  offk : current.regs .int [BK p] "offs_k"=some (fun i => (i.1.val : Int))
  am : current.regs .int [BM p] "offs_am"=some (fun i => (aRow p initial i.1 : Int))
  bn : current.regs .int [BN p] "offs_bn"=some (fun i => (bCol p initial i.1 : Int))

def HasAcc {α : Type} (p : Parameters) (a : CIndex p → α) (s : State α) : Prop :=
  s.regs .real [BM p,BN p] "accumulator"=some a

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (geo : Geometry p s) :
    Context p s 0 (preludeExecution M D p st s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p st s).val := by
  have hm : ((p.M : Int)+((BM p : Int)-1))/(BM p : Int) = (blocksM p : Int) := by
    simpa only [← Int.add_sub_assoc] using cdiv_int p.M (BM p) geo.bm
  have hn : ((p.N : Int)+((BN p : Int)-1))/(BN p : Int) = (blocksN p : Int) := by
    simpa only [← Int.add_sub_assoc] using cdiv_int p.N (BN p) geo.bn
  have hcm : ¬ (p.M : Int)+((BM p : Int)-1)<0 := by have h:=geo.bm; omega
  have hcn : ¬ (p.N : Int)+((BN p : Int)-1)<0 := by have h:=geo.bn; omega
  have hf : (blocksM p : Int) - (firstM p s : Int) = ((blocksM p-firstM p s : Nat) : Int) :=
    (Int.natCast_sub geo.first).symm
  simp only [firstM,perGroup] at hf
  constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,HasAcc]
  all_goals funext i
  all_goals simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right]
  all_goals simp only [hcm,hcn,↓reduceIte,hm,hn,add_zero,zero_add,zero_sub,decide_eq_true_eq]
  all_goals simp only [cast_not_neg,↓reduceIte,←Int.natCast_add,←Int.natCast_mul,
    ←Int.natCast_div,nat_remainder,hf,wide_min]
  all_goals simp only [pidM,pidN,groupSize,perGroup,firstM,aRow,bCol,row,col]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (body p st)
      (current.setReg "k" .int [] (fun _ => (k : Int)))=some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [body,source,typed,Pasted.originalKernel,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.offk,ctx.am,ctx.bn,ha,backend]
     rfl)

private theorem candidate_aValue {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) (k : Nat) (i : AIndex p) :
    (if (k : Int)*(BK p : Int)+(i.2.1.val : Int)<(p.K : Int) then
       read st (s.mem p.A (((aRow p s i.1 : Int)*(p.stride_am : Int)+
         (if (k : Int)*(BK p : Int)+(i.2.1.val : Int)<(p.K : Int)
          then (k : Int)*(BK p : Int)+(i.2.1.val : Int) else 0)*(p.stride_ak : Int)).toNat))
     else M.cast none .real (dtype st) (M.literal none .real 0.0)) = aValue M p st s k i := by
  have hm : ((k : Int)*(BK p : Int)+(i.2.1.val : Int)<(p.K : Int)) ↔ i.2.1.val<p.K-k*BK p := by omega
  by_cases h : (k : Int)*(BK p : Int)+(i.2.1.val : Int)<(p.K : Int)
  · simp only [h,if_true,aValue,hm.mp h]
    simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
    have address : aRow p s i.1*p.stride_am+(k*BK p+i.2.1.val)*p.stride_ak=aOffset p s k i := by
      unfold aOffset
      ring
    rw [address]
  · simp [aValue,h,not_congr hm |>.mp h]

private theorem candidate_bValue {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) (k : Nat) (i : BIndex p) :
    (if (k : Int)*(BK p : Int)+(i.1.val : Int)<(p.K : Int) then
       read st (s.mem p.B (((if (k : Int)*(BK p : Int)+(i.1.val : Int)<(p.K : Int)
          then (k : Int)*(BK p : Int)+(i.1.val : Int) else 0)*(p.stride_bk : Int)+
         (bCol p s i.2.1 : Int)*(p.stride_bn : Int)).toNat))
     else M.cast none .real (dtype st) (M.literal none .real 0.0)) = bValue M p st s k i := by
  have hm : ((k : Int)*(BK p : Int)+(i.1.val : Int)<(p.K : Int)) ↔ i.1.val<p.K-k*BK p := by omega
  by_cases h : (k : Int)*(BK p : Int)+(i.1.val : Int)<(p.K : Int)
  · simp only [h,if_true,bValue,hm.mp h]
    simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
    have address : (k*BK p+i.1.val)*p.stride_bk+bCol p s i.2.1*p.stride_bn=bOffset p s k i := by
      unfold bOffset
      ring
    rw [address]
  · simp [bValue,h,not_congr hm |>.mp h]

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current) :
    let t := (bodyExecution M D p st initial current k a ctx ha).val
    Context p initial (k+1) t ∧ HasAcc p (update M D p st initial k a) t := by
  cases st <;> dsimp only
  all_goals constructor
  all_goals try (constructor <;> simp [bodyExecution,
    ctx.memory,ctx.pm,ctx.pn,ctx.offk,ctx.am,ctx.bn,bop,Broadcast.leftIndex,Broadcast.rightIndex])
  all_goals simp [HasAcc,bodyExecution,update,dtype,backend,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals apply congrArg₂ (fun (av : AIndex p→α) (bv : BIndex p→α) =>
    D.apply (some .fp32) _ _ .real {} (batch := []) av bv a)
  all_goals funext i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_aValue M p .half initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_bValue M p .half initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_aValue M p .bfloat initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_bValue M p .bfloat initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_aValue M p .single initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_bValue M p .single initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_aValue M p .e4m3 initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_bValue M p .e4m3 initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_aValue M p .e5m2 initial k i
  · simpa only [read,dtype,BK,ite_mul,zero_mul] using candidate_bValue M p .e5m2 initial k i

#axiomsClean preludeExecution
#axiomsClean preludeContext
#axiomsClean bodyExecution
#axiomsClean bodyContext
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
