import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.CarryIO
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.WholePair
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

structure Implementation (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (B : Nat) (c : Config) where
  schedule : (NativeIO.io A C arg B c).Inputs → List (Invocation (Term α))
  observe : (o : Fin (NativeIO.io A C arg B c).outputs.length) →
    TileIndex ((NativeIO.io A C arg B c).outputs[o]).shape → Term α → α
  writes : (vals : (NativeIO.io A C arg B c).Inputs) →
    Fin (schedule vals).length → RegionName → Nat → Prop

namespace Implementation
variable {A : Target.Ops α} {C : ScalarABI.Conversions α}
  {arg : ScalarABI.Argument} {B : Nat} {c : Config}

def toIO (p : Implementation A C arg B c) : OrdinaryProgramIO α (Term α) :=
  { NativeIO.io A C arg B c with
    schedule := p.schedule, observe := p.observe, writes := p.writes }

/-- Both complete ordinary executions, actual safety, equality of every active
typed output cell under the declared observers, and both exact raw-cell frames.
No serial schedule runner or assumed successful execution occurs here. -/
def Equivalent (left right : Implementation A C arg B c) : Prop :=
  ∀ vals, IOLayout.nativeDomain B c vals → ∀ initial,
    (NativeIO.io A C arg B c).InputBinding vals initial →
    ∃ l r,
      Nonempty (OrdinaryRun (left.schedule vals) (left.writes vals) initial l) ∧
      Nonempty (OrdinaryRun (right.schedule vals) (right.writes vals) initial r) ∧
      (∀ capacity, (NativeIO.io A C arg B c).InBounds capacity vals →
        left.toIO.Safe vals capacity initial ∧ right.toIO.Safe vals capacity initial) ∧
      (∀ o i, (NativeIO.io A C arg B c).outputMask o vals i →
        mapCell (left.observe o i)
          (l.mem (NativeIO.io A C arg B c).outputs[o].buf
            ((NativeIO.io A C arg B c).write o vals i)) =
        mapCell (right.observe o i)
          (r.mem (NativeIO.io A C arg B c).outputs[o].buf
            ((NativeIO.io A C arg B c).write o vals i))) ∧
      (NativeIO.io A C arg B c).Frame vals initial l ∧
      (NativeIO.io A C arg B c).Frame vals initial r

instance : Spec.ProgramSyntax (Implementation A C arg B c) where
  Statement := Implementation A C arg B c
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some Equivalent
  sameContext := Eq

end Implementation

def original (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (B : Nat) (c : Config) : Implementation A C arg B c where
  schedule := (NativeIO.io A C arg B c).schedule
  observe := (NativeIO.io A C arg B c).observe
  writes := (NativeIO.io A C arg B c).writes

def candidate (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (B : Nat) (c : Config) : Implementation A C arg B c where
  schedule := (CarryIO.io A C arg B c).schedule
  observe := (CarryIO.io A C arg B c).observe
  writes := (CarryIO.io A C arg B c).writes

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.WholePair
