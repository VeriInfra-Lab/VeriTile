import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.IntegerDotSafety

/-! A concrete contextual integer-dot source launch. Launch setup is exactly
Invocation.initial; the added fields contain primitive arithmetic and finite
topology selection only. Execution and source safety remain fixed definitions. -/
namespace VeriTile.Triton.FP.Structural
noncomputable section

structure IntegerDotInvocation (α : Type) extends Invocation α where
  arithmetic : IntegerDot.Arithmetic
  context : IntegerDot.Context

namespace IntegerDotInvocation
def initial {α : Type} (job : IntegerDotInvocation α) (s : State α) : State α :=
  job.toInvocation.initial s

@[simp] theorem initial_mem {α : Type} (job : IntegerDotInvocation α) (s : State α) :
    (job.initial s).mem = s.mem := job.toInvocation.initial_mem s

def run {α : Type} [Inhabited α] (job : IntegerDotInvocation α) (s : State α) :
    Option (State α) :=
  IntegerDot.exec job.backend job.arithmetic job.context job.kernel (job.initial s)

def Safe {α : Type} [Inhabited α] (job : IntegerDotInvocation α)
    (bounds : RegionName → Nat) (s : State α) : Prop :=
  IntegerDot.TraceSafe job.backend job.arithmetic job.context bounds job.kernel (job.initial s)

end IntegerDotInvocation
end
end VeriTile.Triton.FP.Structural
