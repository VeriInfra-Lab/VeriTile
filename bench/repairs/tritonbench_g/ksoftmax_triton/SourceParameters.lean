/-! The three mask modes already admitted by the public host contract.
This source parameter type is shared by the quotation and independent
mathematics; it does not add an input restriction. -/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
inductive MaskMode where
  | absent | qk | bk
  deriving DecidableEq, Repr

def MaskMode.python : MaskMode → Option String
  | .absent => none
  | .qk => some "qk"
  | .bk => some "bk"

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
