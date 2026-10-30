import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.NativeIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.ScheduleEquivalence
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped VeriTile.Spec
noncomputable section

/-- Actual complete original row schedule with its public typed IO. -/
def original {A : Type} (model : Algebra A) (op : Operation) (B Q K D : Nat) :=
  ProgramIO.Implementation.original
    (Interface.programIO model op B Q K D (programs B Q)) (fun _ => True)

/-- The existing reversed candidate changes only the concrete invocation order. -/
def reversed {A : Type} (model : Algebra A) (op : Operation) (B Q K D : Nat) :=
  ProgramIO.Implementation.ofVariant (Interface.reversed model op B Q K D) (fun _ => True)

def assumptions {A : Type} (model : Algebra A) (op : Operation) (B Q K D : Nat) :
    Spec.Assumptions (ProgramIO.Implementation
      (Interface.programIO model op B Q K D (programs B Q)) (fun _ => True)) := []

theorem equivalent {A : Type} [Inhabited A] (model : Algebra A) (op : Operation)
    (B Q K D : Nat) (positive : 0<D) (covered : K≤D) :
    original model op B Q K D ≡[assumptions model op B Q K D] reversed model op B Q K D :=
  ProgramIO.Implementation.of_equivalent (Interface.reversed model op B Q K D)
    (Interface.reverse_equivalent model op B Q K D positive covered)

#print_fp_assumptions equivalent
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.ScheduleEquivalence
