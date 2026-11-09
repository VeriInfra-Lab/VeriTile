import VeriTile.Spec
import VeriTile.Triton.Float.OrdinaryProgramIO
import VeriTile.Meta.StatementAudit

/-! Task-local syntax for complete ordinary-grid implementations. Public
channels, masks and memory windows are fixed by `io`. Implementations contain
only concrete source invocations, row write sets and output observations.
The relation proves successful ordinary runs, access safety, complete active
output-cell equality and both exterior frames. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.OrdinaryEquivalence
open VeriTile.Triton FP.Structural
noncomputable section

structure Implementation {α β : Type} (io : OrdinaryProgramIO α β) where
  schedule : io.Inputs → List (Invocation β)
  writes : (vals : io.Inputs) → Fin (schedule vals).length → RegionName → Nat → Prop
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α

namespace Implementation
variable {α β : Type} {io : OrdinaryProgramIO α β}

def toIO (p : Implementation io) : OrdinaryProgramIO α β :=
  { io with schedule := p.schedule, writes := p.writes, observe := p.observe }

def original (io : OrdinaryProgramIO α β) : Implementation io where
  schedule := io.schedule
  writes := io.writes
  observe := io.observe

def OutputsEqual (left right : Implementation io) (vals : io.Inputs)
    (l r : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (left.observe o j) (l.mem io.outputs[o].buf (io.write o vals j)) =
      mapCell (right.observe o j) (r.mem io.outputs[o].buf (io.write o vals j))

/-- All inputs in the unchanged public domain are admitted. In particular,
successful execution and the output/frame results do not require capacities;
capacities imply safety of each actual source trace. -/
def Equivalent [Inhabited β] (left right : Implementation io) : Prop :=
  ∀ vals initial, io.InputBinding vals initial →
    ∃ l r,
      Nonempty (OrdinaryRun (left.schedule vals) (left.writes vals) initial l) ∧
      Nonempty (OrdinaryRun (right.schedule vals) (right.writes vals) initial r) ∧
      (∀ bounds, io.InBounds bounds vals →
        left.toIO.Safe vals bounds initial ∧ right.toIO.Safe vals bounds initial) ∧
      left.OutputsEqual right vals l r ∧
      io.Frame vals initial l ∧ io.Frame vals initial r

instance programSyntax [Inhabited β] : Spec.ProgramSyntax (Implementation io) where
  Statement := Implementation io
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some Equivalent
  sameContext := Eq

end Implementation
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.OrdinaryEquivalence
