import bench.tritonbench_g.rms_matmul_rbe.Proofs.Prefix
import bench.tritonbench_g.rms_matmul_rbe.Proofs.BodyExecution

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def columns (p : Arguments) : Nat := (p.N+64-1)/64
def programRow (p : Arguments) (s : State α) : Nat := s.pids 1 / columns p
def programCol (p : Arguments) (s : State α) : Nat := s.pids 1 % columns p

theorem signed_columns (n : Nat) :
    (n:Int)/64+(if (n:Int)%64≠0 then 1 else 0)=((n+64-1)/64:Nat) := by
  split_ifs <;> omega

theorem signed_nat_negative (n : Nat) : ¬(n:Int)<0 := by omega

theorem signed_lane16_negative (n lane : Nat) : ¬(n:Int)*16+(lane:Int)<0 := by
  exact not_lt_of_ge (Int.add_nonneg (Int.mul_nonneg (Int.natCast_nonneg _) (by decide))
    (Int.natCast_nonneg _))

theorem signed_lane64_negative (n lane : Nat) : ¬(n:Int)*64+(lane:Int)<0 := by
  exact not_lt_of_ge (Int.add_nonneg (Int.mul_nonneg (Int.natCast_nonneg _) (by decide))
    (Int.natCast_nonneg _))

theorem signed_remainder (n d : Int) : n-d*(n/d)=n%d :=
  (Int.emod_def n d).symm

theorem signed_nat_remainder (n d : Nat) :
    (n:Int)-(d:Int)*((n:Int)/(d:Int))=(n%d:Nat) := by
  simpa only [Int.natCast_emod] using (Int.emod_def (n:Int) (d:Int)).symm

theorem signed_nat_remainder_product (n d : Nat) :
    (n:Int)-(d*(n/d):Nat)=(n%d:Nat) := by
  simpa only [Int.natCast_mul,Int.natCast_ediv] using signed_nat_remainder n d

theorem signed_column_expression (n : Nat) :
    (n:Int)/64+(if (n:Int)-64*((n:Int)/64)=0 then 0 else 1)=((n+64-1)/64:Nat) := by
  split_ifs <;> omega

def initialValues (a : Arithmetic α) (p : Arguments) (s : State α) : LoopValues α where
  keys := fun i => i.1.val
  xptrs := fun i => (p.X,s.pids 0*p.sxb+
    ((programRow p s*16+i.1.val)%p.M)*p.sxm+i.2.1.val*p.sxk)
  wptrs := fun i => (p.W,i.1.val*p.swk+
    ((programCol p s*64+i.2.1.val)%p.N)*p.swn)
  rmsptrs := fun i => (p.RMS,i.2.1.val*p.srms)
  accumulator := fun _ => a.literal .real 0
  moments := fun _ => a.literal .real 0

theorem prefix_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    LoopRegisters (initialValues a p s) (prefixExecution a.model fp8 outputFP8 p s).val := by
  have zeroScientific : (0.0:ℝ)=0 := by norm_num
  constructor
  all_goals
    simp (config := {maxSteps:=20000}) [zeroScientific,signed_columns,prefixExecution,prefixBaseExecution,initialValues,programRow,programCol,columns,
      Arithmetic.model,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      signed_nat_negative,if_false,signed_column_expression]
    simp only [←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      signed_nat_negative,if_false,signed_nat_remainder,Int.toNat_natCast]
    simp only [signed_nat_remainder_product,←Int.natCast_mul _ 16,←Int.natCast_mul _ 64,
      ←Int.natCast_add,signed_nat_negative,if_false,signed_nat_remainder,
      ←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add,Nat.add_sub_assoc]
    try simp only [signed_lane16_negative,signed_lane64_negative,if_false,signed_remainder]
    try norm_cast
    try simp [Nat.add_sub_assoc]

theorem prefix_frame {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.mem=s.mem :=
  prefix_memory M fp8 outputFP8 p s

theorem prefix_coordinates {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.regs .int [] "pid_batch"=some (fun _ => (s.pids 0:Int)) ∧
    (prefixExecution M fp8 outputFP8 p s).val.regs .int [] "pid_m"=some (fun _ => (programRow p s:Int)) ∧
    (prefixExecution M fp8 outputFP8 p s).val.regs .int [] "pid_n"=some (fun _ => (programCol p s:Int)) := by
  simp (config := {maxSteps:=20000}) [signed_columns,prefixExecution,prefixBaseExecution,programRow,programCol,columns,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    constructor <;> funext i <;>
      simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        signed_nat_negative,if_false,signed_column_expression,
        ←Int.natCast_ediv,←Int.natCast_emod,signed_nat_remainder]
  all_goals
    try simp [Nat.add_sub_assoc] at *
    try push_cast
    try omega
  all_goals exact signed_remainder _ _

theorem prefix_scalars {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.regs .real [] "EPS"=s.regs .real [] "EPS" ∧
    (prefixExecution M fp8 outputFP8 p s).val.regs .real [] "THETA"=s.regs .real [] "THETA" := by
  simp (config := {maxSteps:=20000}) [signed_columns,prefixExecution,prefixBaseExecution]

#axiomsClean prefix_values
#axiomsClean prefix_frame
#axiomsClean prefix_coordinates
#axiomsClean prefix_scalars
#specNonCircular initialValues avoiding [prefixExecution,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
