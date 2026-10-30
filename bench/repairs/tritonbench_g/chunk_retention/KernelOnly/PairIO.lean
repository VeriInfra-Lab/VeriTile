import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.IO
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ScaleFirstStageSafety
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket KernelStaging PublicIO
noncomputable section
variable {α : Type} [Inhabited α]

def sourceInvocation (ops : Ops α) (a : Call) (p grid : Nat → Nat) : Invocation (Term α) :=
  launch (engine (withDot (algebra ops.zero) (nativeDot a))) a p grid

def candidateInvocation (ops : Ops α) (a : Call) (p grid : Nat → Nat) : Invocation (Term α) :=
  { sourceInvocation ops a p grid with kernel := candidateKernel a }

/-- The implementation data is one complete actual stage invocation. The
original addressed IO fixes all logical inputs, masks, storage and observer. -/
structure Implementation (ops : Ops α) (literal : ℝ → α) (a : Call) (p grid : Nat → Nat) where
  invocation : Invocation (Term α)

def original (ops : Ops α) (literal : ℝ → α) (a : Call) (p grid : Nat → Nat) :
    Implementation ops literal a p grid := ⟨sourceInvocation ops a p grid⟩
def scaleFirst (ops : Ops α) (literal : ℝ → α) (a : Call) (p grid : Nat → Nat) :
    Implementation ops literal a p grid := ⟨candidateInvocation ops a p grid⟩

/-- Narrow data view of the existing addressed IO: both actual safe, successful
runs, all typed active output cells and both exact raw exterior frames. -/
def Equivalent {ops : Ops α} {literal : ℝ → α} {a : Call} {p grid : Nat → Nat}
    (left right : Implementation ops literal a p grid) : Prop :=
  let io := nativeIO ops literal a p grid
  ∀ xs bounds, io.InBounds bounds xs → ∀ initial, io.InputBinding xs initial →
    ScheduleSafe bounds [left.invocation] initial ∧ ScheduleSafe bounds [right.invocation] initial ∧
    ∃ l r, runSchedule [left.invocation] initial=some l ∧ runSchedule [right.invocation] initial=some r ∧
      (∀ o j, io.outputMask o xs j →
        mapCell (io.observe o j) (l.mem io.outputs[o].buf (io.write o xs j))=
        mapCell (io.observe o j) (r.mem io.outputs[o].buf (io.write o xs j))) ∧
      io.Frame xs initial l ∧ io.Frame xs initial r

instance {ops : Ops α} {literal : ℝ → α} {a : Call} {p grid : Nat → Nat} :
    Spec.ProgramSyntax (Implementation ops literal a p grid) where
  Statement := Implementation ops literal a p grid
  Signature := Unit
  signature := fun _=>()
  body := fun i=>[i]
  structural := some Equivalent
  sameContext := Eq

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PairIO
