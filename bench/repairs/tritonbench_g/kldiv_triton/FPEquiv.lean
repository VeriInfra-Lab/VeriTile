import bench.repairs.tritonbench_g.kldiv_triton.Proofs.GridPair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

def forwardOriginal (m : Tile.Model α) (rows sy st so N B : Nat) (flag : Bool) (red : Nat) (work : List Nat) :=
  ProgramIO.Implementation.original (IO.forward m rows sy st so N B flag red work) (fun _ => True)
def forwardCommuted (m : Tile.Model α) (rows sy st so N B : Nat) (flag : Bool) (red : Nat) (work : List Nat) :=
  ProgramIO.Implementation.ofVariant (IO.forwardCandidate m rows sy st so N B flag red work) (fun _ => True)
def backwardOriginal (m : Tile.Model α) (rows so st N B : Nat) (flag : Bool) (work : List Nat) :=
  ProgramIO.Implementation.original (IO.backward m rows so st N B flag work) (fun _ => True)
def backwardCommuted (m : Tile.Model α) (rows so st N B : Nat) (flag : Bool) (work : List Nat) :=
  ProgramIO.Implementation.ofVariant (IO.backwardCandidate m rows so st N B flag work) (fun _ => True)

/-- Existing signed output-address product candidate, transplanted to the
complete current FP32 source with zero-target/tail repairs and Int64 casts. -/
specification forward_pair (m : Tile.Model α) (rows sy st so N B : Nat) (flag : Bool) (red : Nat)
    (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=outputCount N B red) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    forwardOriginal m rows sy st so N B flag red work ≡[[]] forwardCommuted m rows sy st so N B flag red work :=
  ProgramIO.Implementation.of_equivalent (IO.forwardCandidate m rows sy st so N B flag red work)
    (ProgramPair.forward m rows sy st so N B flag red positive unique work order)

specification backward_pair (m : Tile.Model α) (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=N) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    backwardOriginal m rows so st N B flag work ≡[[]] backwardCommuted m rows so st N B flag work :=
  ProgramIO.Implementation.of_equivalent (IO.backwardCandidate m rows so st N B flag work)
    (ProgramPair.backward m rows so st N B flag positive unique work order)

#axiomsClean forward_pair
#axiomsClean backward_pair
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.FPEquiv
