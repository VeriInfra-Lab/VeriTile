import VeriTile.Triton.Source
namespace VeriTile.Triton.Source
structure SignedRange where
  before : List ComputeStmt
  start : Op .int []
  stop : Op .int []
  stride : Op .nat []
  body : List ComputeStmt
  after : List ComputeStmt

def signedRange? (code : List ComputeStmt) (index : RegName) : Option SignedRange :=
  match code with
  | [] => none
  | st::rest =>
    match st with
    | .forRangeInt name start stop stride body =>
      if name=index then some ⟨[],start,stop,stride,body,rest⟩
      else (signedRange? rest index).map fun r => {r with before:=st::r.before}
    | _ => (signedRange? rest index).map fun r => {r with before:=st::r.before}

def firstAssignment (code : List ComputeStmt) (name : RegName) : Nat :=
  code.findIdx fun statement => match statement with
    | .assign _ _ assigned _ => assigned==name
    | _ => Bool.false


end VeriTile.Triton.Source
