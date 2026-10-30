import Mathlib.Tactic.FinCases
import bench.tritonbench_g.uniform_sampling.Kernels
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.LogicalProgramIO

/-! Canonical field/lane output interface and actual original/candidate calls.
The numerical interpretation is a parameter; all views share the same output
positions, masks and scalar ABI arguments. Connection proofs are in
Proofs/PublicIO.lean. -/
namespace VeriTile.Bench.Repairs.UniformSampling.KernelOffsets

/-- Output position shared by the logical IO and the source offset proofs. -/
def output (pid B f lane : Nat) := (pid*4+f)*B+lane

end VeriTile.Bench.Repairs.UniformSampling.KernelOffsets

namespace VeriTile.Bench.Repairs.UniformSampling.KernelPublic
open VeriTile.Triton FP.Structural
noncomputable section

abbrev io {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat) : ProgramIO α α where
  inputs := []
  outputs := [⟨.real,[4,B],y⟩]
  scratch := []
  buffers := [y]
  read := fun z => Fin.elim0 z
  inputMask := fun z => Fin.elim0 z
  write := fun o => Fin.cases (fun _ j => KernelOffsets.output pid B j.1.val j.2.1.val)
    (fun z => Fin.elim0 z) o
  outputMask := fun o => Fin.cases (fun _ j => KernelOffsets.output pid B j.1.val j.2.1.val<N)
    (fun z => Fin.elim0 z) o
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => [{
    kernel := kernel y N seed offset lower upper B hB
    backend := M
    pids := fun ax => if ax=0 then pid else 0}]
  encode := id
  observe := fun _ _ => id

abbrev variant {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat) :
    ProgramIO.Variant (io M y N seed offset lower upper B hB pid) where
  schedule := fun _ => [{
    kernel := indexCommutedKernel y N seed offset lower upper B hB
    backend := M
    pids := fun ax => if ax=0 then pid else 0}]
  observe := fun _ _ => id

def noInputs {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat)
    (hB : Random.ArangeSizeDomain B) (pid : Nat) :
    (io M y N seed offset lower upper B hB pid).Inputs ≃ PUnit where
  toFun _ := ()
  invFun _ := fun i => Fin.elim0 i
  left_inv := by intro xs; funext i; exact Fin.elim0 i
  right_inv := by intro x; cases x; rfl

def fieldOutputs {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat)
    (hB : Random.ArangeSizeDomain B) (pid : Nat) :
    (io M y N seed offset lower upper B hB pid).Outputs ≃ (Fin 4 → Fin B → α) where
  toFun xs f i := xs 0 (f,i,())
  invFun xs := Fin.cases (fun i => xs i.1 i.2.1) (fun i => Fin.elim0 i)
  left_inv := by intro xs; funext o i; fin_cases o; rcases i with ⟨f,j,u⟩; cases u; rfl
  right_inv := by intro xs; rfl

def logicalIO {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat)
    (hB : Random.ArangeSizeDomain B) (pid : Nat) :
    LogicalProgramIO α α PUnit (Fin 4 → Fin B → α) where
  core := io M y N seed offset lower upper B hB pid
  inputs := noInputs M y N seed offset lower upper B hB pid
  outputs := fieldOutputs M y N seed offset lower upper B hB pid

end
end VeriTile.Bench.Repairs.UniformSampling.KernelPublic

