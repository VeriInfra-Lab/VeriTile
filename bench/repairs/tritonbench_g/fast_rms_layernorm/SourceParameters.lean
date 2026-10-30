import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.FastRmsLayernorm
open VeriTile.Triton

structure Call where
  rows : Nat
  cols : Nat
  block : Nat
  positive : 0<cols
  covered : cols≤block
  blockLimit : block≤65536
  gemma : Bool
  eps : ℝ

def Call.cells (h : Call) : Nat := h.rows*h.cols
end VeriTile.Bench.Repairs.FastRmsLayernorm
