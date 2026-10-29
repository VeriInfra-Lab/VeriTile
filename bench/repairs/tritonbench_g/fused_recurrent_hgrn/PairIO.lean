import bench.repairs.tritonbench_g.fused_recurrent_hgrn.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Candidate invocations and complete source views share recurrentIO. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Pair
open VeriTile.Triton FP.Structural
noncomputable section

def invocation {α : Type} (p : Ops α) (T D BD : Nat) (U S : Bool) (id : Nat × Nat) :=
  IO.invocation p (forwardStrideFolded T D BD U S) id

def variant {α : Type} (p : Ops α) (observe : α → α) (BH T D BDF BDB : Nat) (U S V : Bool) :
    ProgramIO.Variant (IO.recurrentIO p observe BH T D BDF BDB U S V) where
  schedule := fun _ => (IO.programs BH D BDF).map (invocation p T D BDF U S) ++
    (IO.programs BH D BDB).map (IO.invocation p (backward T D BDB U V))
  observe := (IO.recurrentIO p observe BH T D BDF BDB U S V).observe

def original {α : Type} (p : Ops α) (observe : α → α) (BH T D BDF BDB : Nat) (U S V : Bool) :=
  ProgramIO.Implementation.original (IO.recurrentIO p observe BH T D BDF BDB U S V) (fun _ => True)

def strideFolded {α : Type} (p : Ops α) (observe : α → α) (BH T D BDF BDB : Nat) (U S V : Bool) :=
  ProgramIO.Implementation.ofVariant (variant p observe BH T D BDF BDB U S V) (fun _ => True)

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Pair
