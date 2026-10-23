import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Contract

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Mathematical
noncomputable section
open scoped BigOperators

/-- Arbitrary mathematical arrays; no finite floating-word decoder restricts
this domain. Initial output is independent and may contain any values. -/
structure Inputs where
  a : Nat → ℝ
  dout : Nat → ℝ
  residual : Nat → ℝ
  output : Nat → ℝ

def tensorIndex (p : Metadata) (b s g n : Nat) := ((b*seqlen p+s)*groups p+g)*width p+n
def doutIndex (p : Metadata) (b c g k r : Nat) := (((b*chunks p+c)*groups p+g)*chunk p+k)*chunk p+r

def chunkProduct (p : Metadata) (d : Inputs) (b c g r n : Nat) : ℝ :=
  (∑ k : Fin (chunk p),if k.val<min (chunk p) (seqlen p-c*chunk p) then
    d.dout (doutIndex p b c g k.val r)*d.a (tensorIndex p b (c*chunk p+k.val) g n) else 0)+
  if p.hasResidual then d.residual (tensorIndex p b (c*chunk p+r) g n) else 0

/-- Complete independent Doutᵀ @ A plus optional residual at every valid
sequence coordinate, including the shortened final chunk. -/
def product (p : Metadata) (d : Inputs) (b s g n : Nat) : ℝ :=
  chunkProduct p d b (s/chunk p) g (s%chunk p) n
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Mathematical
