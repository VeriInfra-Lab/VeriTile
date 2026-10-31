import bench.tritonbench_g.triton_matmul.Proofs.SourceMetadata
import bench.tritonbench_g.triton_matmul.Proofs.SourceArithmetic
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 700000
set_option linter.unusedSimpArgs false

def ceilCount (x b : Nat) := x/b+(if x%b=0 then 0 else 1)

private theorem ceiling_expression (x b : Nat) :
    (if (x : Int)<0 then -(-(x : Int)/(b : Int)) else (x : Int)/(b : Int))+
      (if (x : Int)-(if (x : Int)<0 then -((b : Int)*(-(x : Int)/(b : Int)))
         else (b : Int)*((x : Int)/(b : Int)))=0 then 0 else 1)=
      ((x : Int)/(b : Int)+(if x%b=0 then 0 else 1)) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_emod]
  by_cases hz:x%b=0
  · simp only [ceilCount,hz,Int.natCast_zero,↓reduceIte,Int.natCast_add,Int.natCast_ediv]
  · have hi : ((x%b : Nat) : Int)≠0 := by exact_mod_cast hz
    simp only [ceilCount,hz,hi,↓reduceIte,Int.natCast_add,Int.natCast_ediv,Int.natCast_one]

theorem count_m {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.regs .int [] "num_pid_m"=
      some (fun _=>(ceilCount p.M (BM p) : Int)) := by
  cases st <;> simp [preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.ofNat_eq_natCast,ceilCount]
  all_goals funext i
  all_goals simpa only [bop,Broadcast.leftIndex,Broadcast.rightIndex] using ceiling_expression p.M (BM p)

theorem count_n {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.regs .int [] "num_pid_n"=
      some (fun _=>(ceilCount p.N (BN p) : Int)) := by
  cases st <;> simp [preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.ofNat_eq_natCast,ceilCount]
  all_goals funext i
  all_goals simpa only [bop,Broadcast.leftIndex,Broadcast.rightIndex] using ceiling_expression p.N (BN p)

theorem count_k {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.regs .int [] "blocks_k"=
      some (fun _=>(ceilCount p.K (BK p) : Int)) := by
  cases st <;> simp [preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.ofNat_eq_natCast,ceilCount]
  all_goals funext i
  all_goals simpa only [bop,Broadcast.leftIndex,Broadcast.rightIndex] using ceiling_expression p.K (BK p)

#axiomsClean count_m
#axiomsClean count_n
#axiomsClean count_k
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
