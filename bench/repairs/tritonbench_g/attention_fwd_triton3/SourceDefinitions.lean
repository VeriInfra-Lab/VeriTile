import bench.tritonbench_g.attention_fwd_triton3.SourceParameters

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Triton

namespace Geometry
/-- Metadata-only four-dimensional view. Allocation offsets are measured from
its data pointer; the public Tensor contract supplies storage-offset transport. -/
structure Strides where
  batch : Nat
  head : Nat
  row : Nat
  dim : Nat
  deriving DecidableEq

/-- Integer ceil-div. The host float spelling is exact for the admitted
signed-INT32 context lengths; its equality is a public host scalar boundary. -/
def tiles (n b : Nat) : Nat := (n+b-1)/b

end Geometry

namespace Host
open Geometry
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

structure Call where
  Z : Nat
  H : Nat
  HK : Nat
  NQ : Nat
  NK : Nat
  D : Nat
  BM : Nat
  BN : Nat
  q : Strides
  k : Strides
  v : Strides
  o : Strides
  scale : ℝ
  init : Nat
  finish : Nat
  window : Nat
  complement : Nat
  offset : Int
  size : Int

def Call.rounded (c : Call) := tiles c.NQ 64*64

def Call.params (c : Call) : RealPrologue.Parameters where
  heads := c.H
  kvHeads := c.HK
  strides := ⟨c.q.batch,c.q.head,c.k.batch,c.k.head,c.v.batch,c.v.head,c.o.batch,c.o.head⟩
  layout := ⟨"Q","K","V","M","O","L",c.NQ,c.rounded,c.NK,
    c.q.row,c.q.dim,c.k.dim,c.k.row,c.v.row,c.v.dim,c.o.row,c.o.dim⟩
  scale := c.scale
  init := c.init
  evenM := if c.NQ%c.BM=0 then 1 else 0
  evenN := if c.NK%c.BN=0 then 1 else 0
  window := c.window
  complement := c.complement
  offset := c.offset
  size := c.size

end
end Host
end VeriTile.Bench.Repairs.AttentionFwdTriton3
