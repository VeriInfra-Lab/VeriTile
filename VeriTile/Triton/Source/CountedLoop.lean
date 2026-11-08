import VeriTile.Triton.Source

namespace VeriTile.Triton.Source
/-- Locate the actual counted-loop constructor without rewriting its body or
its surrounding code. Bounds are returned as operations for proof-side views. -/
def countedLoop? (code : List ComputeStmt) (index : RegName) : Option DynamicRange :=
  match code with
  | [] => none
  | st :: rest =>
    match st with
    | .forLoop name stop body =>
      if name = index then some ⟨[], .constNat 0, .constNat stop, .constNat 1, body, rest⟩
      else (countedLoop? rest index).map fun r => { r with before := st :: r.before }
    | _ => (countedLoop? rest index).map fun r => { r with before := st :: r.before }
end VeriTile.Triton.Source
