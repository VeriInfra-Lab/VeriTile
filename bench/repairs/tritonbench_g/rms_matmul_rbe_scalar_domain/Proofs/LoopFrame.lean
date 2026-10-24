import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.LoopState

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def headerNames : List RegName := ["EPS","THETA","pid_batch","pid_m","pid_n"]

def HeaderPreserved (initial current : State α) : Prop :=
  ∀ d sh name,name∈headerNames → current.regs d sh name=initial.regs d sh name

theorem HeaderPreserved.refl (s : State α) : HeaderPreserved s s := fun _ _ _ _ => rfl
theorem HeaderPreserved.trans (initial mid last : State α)
    (h : HeaderPreserved initial mid) (k : HeaderPreserved mid last) :
    HeaderPreserved initial last := fun d sh name hn => (k d sh name hn).trans (h d sh name hn)

theorem body_header {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    HeaderPreserved s (bodyExecution a fp8 outputFP8 p s v h bytes t).val := by
  intro d sh name hn
  simp [headerNames] at hn
  rcases hn with rfl|rfl|rfl|rfl|rfl
  all_goals cases fp8 <;> cases outputFP8 <;> simp [bodyExecution]

#axiomsClean body_header
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
