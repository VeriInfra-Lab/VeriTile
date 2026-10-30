import bench.tritonbench_g.bmm_chunk_fwd.Contract

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Mathematical
noncomputable section
/-- The mathematical carrier is independent of finite floating encodings. All
input/output Real cells are arbitrary. Sequence IDs retain signed INT64 values. -/
structure Inputs where
  a : Nat → ℝ
  b : Nat → ℝ
  seq : Nat → BitVec 64
  output : Nat → ℝ

def inputIndex (p : Metadata) (b r g k : Nat) :=
  ((b*(shape p).seqlen+r)*(shape p).groups+g)*(shape p).K+k

/-- Complete returned-tensor target: a chunked A*transpose(B) contraction,
optional signed-ID equality, zero padding, and an exact lower triangle in causal
mode. No execution, selected configuration, or initial output occurs here. -/
def product (p : Metadata) (x : Inputs) (b ch g r n : Nat) : ℝ :=
  if p.causal=Bool.true ∧ r<n then 0 else
  if r<min (shape p).chunk ((shape p).seqlen-ch*(shape p).chunk) ∧
      n<min (shape p).chunk ((shape p).seqlen-ch*(shape p).chunk) ∧
      (p.hasSeq=Bool.false ∨ (x.seq (b*(shape p).seqlen+ch*(shape p).chunk+r)).toInt=
        (x.seq (b*(shape p).seqlen+ch*(shape p).chunk+n)).toInt)
  then ∑ k : Fin (shape p).K,
    x.a (inputIndex p b (ch*(shape p).chunk+r) g k)*
    x.b (inputIndex p b (ch*(shape p).chunk+n) g k)
  else 0
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Mathematical
