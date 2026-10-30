import bench.tritonbench_g.matmul_triton1.Proofs.CheckedHost
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.LogicalInputs
open VeriTile.Triton FP.Structural

open SharedContract TensorBinding
noncomputable section
variable {α : Type} [Inhabited α]

def precisionA (P : Core.Primitives α) (a : TensorBinding.Tensor) (la : Laws a)
    (read : Nat → Word) (decode : Word → α) (BK r block : Nat) (e : Fin BK) : α :=
  if e.val < columns a.metadata-block*BK then
    decode (VeriTile.Torch.FP32.logicalValue read a la (r*columns a.metadata+(block*BK+e.val)))
  else P.zero

def precisionB (P : Core.Primitives α) (b : TensorBinding.Tensor) (lb : Laws b)
    (read : Nat → Word) (decode : Word → α) (K BK c block : Nat) (e : Fin BK) : α :=
  if e.val < K-block*BK then
    decode (VeriTile.Torch.FP32.logicalValue read b lb ((block*BK+e.val)*columns b.metadata+c))
  else P.zero

def precisionAccum (P : Core.Primitives α) (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (read : Nat → Word) (decode : Word → α) (BK r c : Nat) : Nat → α
  | 0 => P.zero
  | block+1 => P.dot (precisionA P a la read decode BK r block)
      (precisionB P b lb read decode (columns a.metadata) BK c block)
      (precisionAccum P a b la lb read decode BK r c block)

def precisionReference (P : Core.Primitives α) (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (read : Nat → Word) (decode : Word → α) (BK r c : Nat) : α :=
  precisionAccum P a b la lb read decode BK r c (FiniteIndex.ceil (columns a.metadata) BK)

def realExpected (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (read : Nat → Word) (decode : Word → ℝ) (r c : Nat) : ℝ :=
  RealSum.sum (fun k =>
    decode (VeriTile.Torch.FP32.logicalValue read a la (r*columns a.metadata+k)) *
    decode (VeriTile.Torch.FP32.logicalValue read b lb (k*columns b.metadata+c))) (columns a.metadata)

theorem scalar_inputs (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (ra rb : RegionName) (default : Bool)
    (h : SharedContract.InputContract Bool.true default a.metadata b.metadata)
    (read : Nat → Word) (decode : Word → α) (initial : State α)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read decode initial)
    (r c k : Nat) (hr : r < rows a.metadata) (hc : c < columns b.metadata)
    (hk : k < columns a.metadata) :
    (initial.mem ra (r*columns a.metadata+k)).read .real =
      decode (VeriTile.Torch.FP32.logicalValue read a la (r*columns a.metadata+k)) ∧
    (initial.mem rb (k*columns b.metadata+c)).read .real =
      decode (VeriTile.Torch.FP32.logicalValue read b lb (k*columns b.metadata+c)) := by
  constructor
  · apply TensorBinding.logical_left a b la lb Bool.true default h ra rb read decode initial bound
    rw [matrix_numel a.metadata h.rank.1]
    exact FiniteIndex.linear_bound _ _ _ _ hr hk
  · apply TensorBinding.logical_right a b la lb Bool.true default h ra rb read decode initial bound
    rw [matrix_numel b.metadata h.rank.2,←h.dimensions]
    exact FiniteIndex.linear_bound _ _ _ _ hk hc

theorem real_reference (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (ra rb : RegionName) (default : Bool)
    (h : SharedContract.InputContract Bool.true default a.metadata b.metadata)
    (read : Nat → Word) (decode : Word → ℝ) (initial : State ℝ)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read decode initial)
    (c : Host.Tensor) (i : Host.Index (InputAdapter.arguments a b ra rb default)) :
    Core.realMatmul (Host.parameters (InputAdapter.arguments a b ra rb default) c) initial i.1.val i.2.val =
      realExpected a b la lb read decode i.1.val i.2.val := by
  unfold Core.realMatmul realExpected
  apply RealSum.sum_congr
  intro k hk
  have values := scalar_inputs a b la lb ra rb default h read decode initial bound
    i.1.val i.2.val k i.1.isLt i.2.isLt hk
  simpa only [Core.realProduct,Host.parameters,InputAdapter.arguments,InputAdapter.tensor,Nat.mul_one,values.1,values.2]

theorem precision_reference (P : Core.Primitives α) (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (ra rb : RegionName) (default : Bool)
    (h : SharedContract.InputContract Bool.true default a.metadata b.metadata)
    (read : Nat → Word) (decode : Word → α) (initial : State α)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read decode initial)
    (c : Host.Tensor) (i : Host.Index (InputAdapter.arguments a b ra rb default)) :
    Core.logical P (Host.parameters (InputAdapter.arguments a b ra rb default) c) initial i.1.val i.2.val =
      precisionReference P a b la lb read decode
        (InputAdapter.arguments a b ra rb default).config.blockInner i.1.val i.2.val := by
  have inductionStep : ∀ n,
      Core.logicalAccum P (Host.parameters (InputAdapter.arguments a b ra rb default) c) initial i.1.val i.2.val n =
        precisionAccum P a b la lb read decode
          (InputAdapter.arguments a b ra rb default).config.blockInner i.1.val i.2.val n := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      simp only [Core.logicalAccum,precisionAccum,ih]
      congr 1
      · funext e
        unfold Core.logicalA precisionA
        simp only [Host.parameters,InputAdapter.arguments,InputAdapter.tensor,Core.inner,Nat.mul_one]
        split
        · rename_i active
          exact (scalar_inputs a b la lb ra rb default h read decode initial bound
            i.1.val i.2.val _ i.1.isLt i.2.isLt (by omega)).1
        · rfl
      · funext e
        unfold Core.logicalB precisionB
        simp only [Host.parameters,InputAdapter.arguments,InputAdapter.tensor,Core.inner,Nat.mul_one]
        split
        · rename_i active
          exact (scalar_inputs a b la lb ra rb default h read decode initial bound
            i.1.val i.2.val _ i.1.isLt i.2.isLt (by omega)).2
        · rfl
  exact inductionStep _

#specNonCircular realExpected avoiding [CheckedHost.call,Core.source,FP.Structural.exec]
#specNonCircular precisionReference avoiding [CheckedHost.call,Core.source,FP.Structural.exec]
#axiomsClean real_reference
#axiomsClean precision_reference
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.LogicalInputs
