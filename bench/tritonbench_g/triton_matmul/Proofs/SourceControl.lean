import bench.tritonbench_g.triton_matmul.Proofs.SourceCounts
import bench.tritonbench_g.triton_matmul.Proofs.SourceControlExecution
import bench.tritonbench_g.triton_matmul.Proofs.SourceArithmetic
import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.SourceControl
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 700000
set_option linter.unusedSimpArgs false

def rows (p : Parameters) := ceilCount p.M (BM p)
def columns (p : Parameters) := ceilCount p.N (BN p)
def groupPrograms (p : Parameters) := p.GROUP_SIZE_M*columns p

def first {α : Type} (p : Parameters) (s : State α) := s.pids 0/groupPrograms p*p.GROUP_SIZE_M
def groupSize {α : Type} (p : Parameters) (s : State α) := min (rows p-first p s) p.GROUP_SIZE_M
def tileM {α : Type} (p : Parameters) (s : State α) := first p s+s.pids 0%groupSize p s
def tileN {α : Type} (p : Parameters) (s : State α) := (s.pids 0%groupPrograms p)/groupSize p s

theorem wide_ceil (x b : Nat) :
    (if (x : Int)<0 then -(-(x : Int)/(b : Int)) else (x : Int)/(b : Int))+
      (if (x : Int)-(if (x : Int)<0 then -((b : Int)*(-(x : Int)/(b : Int)))
         else (b : Int)*((x : Int)/(b : Int)))=0 then 0 else 1)=
      (ceilCount x b : Int) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_emod]
  by_cases hz:x%b=0
  · simp only [ceilCount,hz,Int.natCast_zero,↓reduceIte,Int.natCast_add,Int.natCast_ediv]
  · have hi : ((x%b : Nat) : Int)≠0 := by exact_mod_cast hz
    simp only [ceilCount,hz,hi,↓reduceIte,Int.natCast_add,Int.natCast_ediv,Int.natCast_one]

theorem wide_div (x b : Nat) :
    (if (x : Int)<0 then -(-(x : Int)/(b : Int)) else (x : Int)/(b : Int))=(x/b : Nat) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte,Int.natCast_ediv]

theorem wide_mod (x b : Nat) :
    (x : Int)-(if (x : Int)<0 then -((b : Int)*(-(x : Int)/(b : Int)))
      else (b : Int)*((x : Int)/(b : Int)))=(x%b : Nat) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_emod]

theorem wide_min (x y : Nat) :
    (if (x : Int)<(y : Int) then (x : Int) else (y : Int))=(min x y : Nat) := by
  by_cases h:x<y
  · simp [h,show (x : Int)<(y : Int) by exact_mod_cast h,Nat.min_eq_left (Nat.le_of_lt h)]
  · simp [h,show ¬(x : Int)<(y : Int) by exact_mod_cast h,Nat.min_eq_right (Nat.le_of_not_gt h)]

theorem prelude_group_programs {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (prefixExecution M D p st s).val.regs .int [] "num_pid_in_group"=
      some (fun _=>(groupPrograms p : Int)) := by
  cases st <;> simp [prefixExecution]
  all_goals funext i
  all_goals simpa only [groupPrograms,columns,Int.natCast_mul,bop,Broadcast.leftIndex,Broadcast.rightIndex] using
    congrArg (fun v : Int=>(p.GROUP_SIZE_M : Int)*v) (wide_ceil p.N (BN p))

theorem prelude_first {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (prefixExecution M D p st s).val.regs .int [] "first_pid_m"=
      some (fun _=>(first p s : Int)) := by
  cases st <;> simp [prefixExecution]
  all_goals funext i
  all_goals try simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rw [wide_ceil]
  all_goals simp only [←Int.natCast_mul]
  all_goals rw [wide_div]
  all_goals simp only [first,groupPrograms,columns,Int.natCast_mul]

theorem prelude_group_size {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (hf : first p s≤rows p) :
    (prefixExecution M D p st s).val.regs .int [] "group_size_m"=
      some (fun _=>(groupSize p s : Int)) := by
  cases st <;> simp [prefixExecution]
  all_goals funext i
  all_goals try simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp_rw [wide_ceil]
  all_goals simp only [←Int.natCast_mul]
  all_goals try simp_rw [wide_div]
  all_goals try simp only [←Int.natCast_mul]
  all_goals change (if (rows p : Int)-(first p s : Int)<(p.GROUP_SIZE_M : Int)
    then (rows p : Int)-(first p s : Int) else (p.GROUP_SIZE_M : Int))=(groupSize p s : Int)
  all_goals rw [←Int.natCast_sub hf]
  all_goals exact wide_min _ _

theorem first_bound {α : Type} (p : Parameters) (s : State α)
    (hp : s.pids 0<rows p*columns p) : first p s<rows p := by
  by_contra h
  have hr:rows p≤first p s:=by omega
  have hh:rows p*columns p≤s.pids 0 := calc
    rows p*columns p≤first p s*columns p := Nat.mul_le_mul_right _ hr
    _=s.pids 0/(p.GROUP_SIZE_M*columns p)*(p.GROUP_SIZE_M*columns p) := by
      simp only [first,groupPrograms,Nat.mul_assoc]
    _≤s.pids 0:=Nat.div_mul_le_self _ _
  omega

theorem prelude_tileM {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (hf : first p s≤rows p) :
    (prefixExecution M D p st s).val.regs .int [] "pid_m"=
      some (fun _=>(tileM p s : Int)) := by
  cases st <;> simp [prefixExecution]
  all_goals funext i
  all_goals try simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp_rw [wide_ceil]
  all_goals simp only [not_lt.mpr (Int.natCast_nonneg (s.pids 0)),↓reduceIte,
    ←Int.natCast_ediv,←Int.natCast_mul,add_zero,zero_add]
  all_goals change (first p s : Int)+((s.pids 0 : Int)-
    (if (rows p : Int)-(first p s : Int)<(p.GROUP_SIZE_M : Int)
      then (rows p : Int)-(first p s : Int) else (p.GROUP_SIZE_M : Int))*
    ((s.pids 0 : Int)/(if (rows p : Int)-(first p s : Int)<(p.GROUP_SIZE_M : Int)
      then (rows p : Int)-(first p s : Int) else (p.GROUP_SIZE_M : Int))))=(tileM p s : Int)
  all_goals rw [←Int.natCast_sub hf,wide_min,←Int.emod_def,←Int.natCast_emod]
  all_goals exact (Int.natCast_add _ _).symm

theorem prelude_tileN {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (hf : first p s≤rows p) :
    (prefixExecution M D p st s).val.regs .int [] "pid_n"=
      some (fun _=>(tileN p s : Int)) := by
  cases st <;> simp [prefixExecution]
  all_goals funext i
  all_goals try simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp_rw [wide_ceil]
  all_goals simp only [not_lt.mpr (Int.natCast_nonneg (s.pids 0)),↓reduceIte,
    ←Int.natCast_ediv,←Int.natCast_mul,add_zero,zero_add]
  all_goals
    have hz : ¬ (s.pids 0 : Int)<(groupPrograms p*(s.pids 0/groupPrograms p) : Nat) := by
      exact_mod_cast not_lt.mpr (Nat.mul_div_le (s.pids 0) (groupPrograms p))
  all_goals change (if (s.pids 0 : Int)<(groupPrograms p*(s.pids 0/groupPrograms p) : Nat) then _ else _)=_
  all_goals rw [if_neg hz]
  all_goals change ((s.pids 0 : Int)-(groupPrograms p*(s.pids 0/groupPrograms p) : Nat))/
    (if (rows p : Int)-(first p s : Int)<(p.GROUP_SIZE_M : Int)
      then (rows p : Int)-(first p s : Int) else (p.GROUP_SIZE_M : Int))=(tileN p s : Int)
  all_goals
    have hm : (s.pids 0 : Int)-(groupPrograms p*(s.pids 0/groupPrograms p) : Nat)=
      (s.pids 0%groupPrograms p : Nat) := by
      rw [Int.natCast_mul,Int.natCast_ediv,←Int.emod_def,←Int.natCast_emod]
  all_goals rw [hm,←Int.natCast_sub hf,wide_min,←Int.natCast_ediv]
  all_goals rfl

theorem prefix_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (prefixExecution M D p st s).val.mem=s.mem := by
  cases st <;> simp [prefixExecution]

theorem prefix_metadata {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    Metadata p (prefixExecution M D p st s).val := by
  cases st <;> simp [Metadata,prefixExecution]

theorem coordinates_eq (p : Parameters) (s : State α) (hm : 0<BM p) (hn : 0<BN p) :
    rows p=blocksM p ∧ columns p=blocksN p ∧ tileM p s=pidM p s ∧ tileN p s=pidN p s := by
  have hr : rows p=blocksM p:=SourceArithmetic.ceiling _ _ hm
  have hc : columns p=blocksN p:=SourceArithmetic.ceiling _ _ hn
  exact ⟨hr,hc,by simp only [tileM,first,groupPrograms,groupSize,pidM,firstM,perGroup,VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue.groupSize,hr,hc],
    by simp only [tileN,groupPrograms,groupSize,first,pidN,perGroup,VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue.groupSize,firstM,hr,hc]⟩

#axiomsClean prefix_memory
#axiomsClean prefix_metadata
#axiomsClean coordinates_eq

#axiomsClean prelude_tileN

#axiomsClean prelude_tileM

#axiomsClean first_bound
#axiomsClean prelude_first
#axiomsClean prelude_group_size
#axiomsClean wide_ceil
#axiomsClean wide_div
#axiomsClean wide_mod
#axiomsClean wide_min
#axiomsClean prelude_group_programs
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.SourceControl
