import bench.tritonbench_g.fast_layernorm.Proofs.KernelOutputs

namespace VeriTile.Bench.Repairs.FastLayernorm.RawValues
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def value (u : State α) (r : RegionName) (i : Nat) : α := (u.mem r i).read .real

theorem raw (u : State α) (r : RegionName) (stride B : Nat) :
    KernelRows.raw u r stride B=(fun j => value u r (u.pids 0*stride+j.val)) := rfl

def affineValues (M : Algebra α) (rows N : Nat) (u : State α) (j : Nat) : α × α :=
  KernelReference.affineCarry M (decide (j<N))
    (fun row => value u "x" (row*N+(if j<N then j else 0)))
    (fun row => value u "dy" (row*N+(if j<N then j else 0)))
    (value u "mu") (value u "r") rows

theorem carry (M : Algebra α) (rows N : Nat) (u : State α) (j : Fin 128) :
    KernelAffine.carry M N (u.pids 0) u rows (j,PUnit.unit)=affineValues M rows N u (u.pids 0*128+j.val) := rfl
end
end VeriTile.Bench.Repairs.FastLayernorm.RawValues
