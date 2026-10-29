import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Prefix
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.BodyExecution

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def columns (p : Arguments) : Nat := (p.N+64-1)/64
def programRow (p : Arguments) (s : State α) : Nat := s.pids 1 / columns p
def programCol (p : Arguments) (s : State α) : Nat := s.pids 1 % columns p

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
  cases fp8 <;> cases outputFP8
  all_goals
    constructor <;> simp [prefixExecution,initialValues,programRow,programCol,columns,
      Arithmetic.model,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      funext i
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem prefix_frame {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.mem=s.mem := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem prefix_coordinates {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.regs .nat [] "pid_batch"=some (fun _ => s.pids 0) ∧
    (prefixExecution M fp8 outputFP8 p s).val.regs .nat [] "pid_m"=some (fun _ => programRow p s) ∧
    (prefixExecution M fp8 outputFP8 p s).val.regs .nat [] "pid_n"=some (fun _ => programCol p s) := by
  cases fp8 <;> cases outputFP8 <;>
    simp [prefixExecution,programRow,programCol,columns,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    constructor <;> funext i <;>
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem prefix_scalars {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.regs .real [] "EPS"=s.regs .real [] "EPS" ∧
    (prefixExecution M fp8 outputFP8 p s).val.regs .real [] "THETA"=s.regs .real [] "THETA" := by
  cases fp8 <;> cases outputFP8 <;> simp [prefixExecution]

#axiomsClean prefix_values
#axiomsClean prefix_frame
#axiomsClean prefix_coordinates
#axiomsClean prefix_scalars
#specNonCircular initialValues avoiding [prefixExecution,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
