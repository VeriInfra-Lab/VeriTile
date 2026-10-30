import bench.repairs.tritonbench_g.spinning_lock_reduction.IO
import VeriTile.Spec

/-! Concrete source views with the fixed public P/C/lock interface. The source
reader computes each decoded controller; implementations contain source data,
not a runner or a proof callback. The structural relation requires completed
safe runs, equal complete final states, observed outputs and exterior frames.
-/
namespace VeriTile.Bench.Repairs.SpinningLock.ProgramView
open VeriTile Triton VeriTile.Bench.TritonBenchG.SpinningLock
noncomputable section
variable {V : Type} {h : HostArgs} {count : Nat} {stage : Phased.Stage}

structure Implementation (b : Backend V) (h : HostArgs) (count : Nat) (stage : Phased.Stage) where
  source : List SourceReader.Statement

variable {b : Backend V}

def Implementation.io (p : Implementation b h count stage) :
    Concurrent.EventProgramIO (List SourceReader.Statement) V V V :=
  { IO.io V h count stage with source := p.source, decoded := ConcurrentSource.decode h stage p.source }

def Equivalent (b : Backend V) (left right : Implementation b h count stage) : Prop :=
  ∀ input, IO.domain b.zero h count stage input → ∀ before,
    (IO.io V h count stage).InputBinding input before →
    ∃ a z events final,
      left.io.decoded = some a ∧ right.io.decoded = some z ∧
      a.Runs (ConcurrentSource.backend b) before events final ∧
      z.Runs (ConcurrentSource.backend b) before events final ∧
      (∀ bounds, left.io.InBounds bounds → Concurrent.TapeSafe bounds events) ∧
      (∀ bounds, right.io.InBounds bounds → Concurrent.TapeSafe bounds events) ∧
      left.io.OutputBinding (IO.mathematical b.zero h count stage input) final ∧
      right.io.OutputBinding (IO.mathematical b.zero h count stage input) final ∧
      left.io.Frame before final ∧ right.io.Frame before final

instance (b : Backend V) : Spec.ProgramSyntax (Implementation b h count stage) where
  Statement := Implementation b h count stage
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some (Equivalent b)
  sameContext := Eq

end
end VeriTile.Bench.Repairs.SpinningLock.ProgramView
