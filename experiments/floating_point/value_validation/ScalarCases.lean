import VeriTile.Triton.Float.ScalarOps
import Lean
open VeriTile.Triton.FP Lean

/-- Reproducible finite probe inputs; this executable emits model results and
is not a proof that any native implementation matches them. -/
def edgeWords : List Nat := [0, 0x80000000, 1, 0x80000001, 0x007fffff,
  0x00800000, 0x3f800000, 0xbf800000, 0x33000000, 0x33800000,
  0x3f800001, 0x3f7fffff, 0x40000000, 0x40400000, 0x7f7fffff,
  0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001,
  0xffc12345, 0xff800001]

def pairCases : List (Nat × Nat) :=
  edgeWords.flatMap fun a => [0, 0x80000000, 1, 0x3f800000, 0x33000000,
    0x40400000, 0x7f7fffff, 0x7f800000, 0x7fc00001].map fun b => (a,b)

def binaryCase (op : String) (f : Value .fp32 → Value .fp32 → Value .fp32)
    (pair : Nat × Nat) : Json :=
  Json.mkObj [("op",toJson op), ("a",toJson pair.1), ("b",toJson pair.2),
    ("expected",toJson (f (Value.ofBits .fp32 pair.1) (Value.ofBits .fp32 pair.2)).raw)]

def main : IO Unit := do
  let ops := [("add",add Config.ieeeValue), ("sub",sub Config.ieeeValue),
    ("mul",mul Config.ieeeValue), ("div",div Config.ieeeValue)]
  let rows := (ops.flatMap fun (op,f) => pairCases.map (binaryCase op f)) ++
    edgeWords.map (fun a => Json.mkObj [("op",toJson ("neg" : String)),
      ("a",toJson a),("b",toJson (0 : Nat)),
      ("expected",toJson (Value.ofBits .fp32 a).neg.raw)]) ++
    edgeWords.map (fun a => Json.mkObj [("op",toJson ("abs" : String)),
      ("a",toJson a),("b",toJson (0 : Nat)),
      ("expected",toJson (Value.ofBits .fp32 a).abs.raw)])
  IO.println (Json.mkObj [("schema",toJson (1 : Nat)),
    ("model",toJson ("VeriTile.Triton.FP.ScalarOps; Config.ieeeValue" : String)),
    ("arithmetic_nan_policy",toJson ("canonical positive quiet NaN" : String)),
    ("neg_nan_policy",toJson ("flip sign and preserve all payload bits" : String)),
    ("abs_nan_policy",toJson ("clear sign and preserve all payload bits" : String)),
    ("cases",toJson rows)]).compress
