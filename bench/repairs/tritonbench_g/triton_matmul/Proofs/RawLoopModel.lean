import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawPreludeContext
import bench.tritonbench_g.triton_matmul.Proofs.FPReference

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

def innerMask (p : Parameters) (k : Nat) (i : Fin (BK p)) : Bool :=
  decide ((i.val : Int) < (p.K : Int) - (k : Int) * (BK p : Int))
def innerOffset (p : Parameters) (k : Nat) (i : Fin (BK p)) : Int :=
  (if innerMask p k i then (k : Int) * (BK p : Int) else 0) +
    (if innerMask p k i then (i.val : Int) else 0)

def addressA {α : Type} (p : Parameters) (s : State α) (k : Nat) (i : AIndex p) : Nat :=
  (offsetM p s i.1 * (p.stride_am : Int) + innerOffset p k i.2.1 * (p.stride_ak : Int)).toNat
def addressB {α : Type} (p : Parameters) (s : State α) (k : Nat) (i : BIndex p) : Nat :=
  (innerOffset p k i.1 * (p.stride_bk : Int) + offsetN p s i.2.1 * (p.stride_bn : Int)).toNat

def loadA {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage)
    (s : State α) (k : Nat) (i : AIndex p) : α :=
  if innerMask p k i.2.1 then FPValue.read st (s.mem p.A (addressA p s k i))
  else M.cast none .real (dtype st) (M.literal none .real 0.0)
def loadB {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage)
    (s : State α) (k : Nat) (i : BIndex p) : α :=
  if innerMask p k i.1 then FPValue.read st (s.mem p.B (addressB p s k i))
  else M.cast none .real (dtype st) (M.literal none .real 0.0)

def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (k : Nat) (acc : CIndex p → α) : CIndex p → α :=
  D.apply (some .fp32) (dtype st) (dtype st) .real {} (batch := [])
    (loadA M p st s k) (loadB M p st s k) acc

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
