import VeriTile.Triton.Core.Ast

namespace VeriTile.Triton.Source

/-- A view of an actual top-level while statement, with its unchanged context. -/
structure WhileLoop where
  before : List ComputeStmt
  condition : Op .bool []
  body : List ComputeStmt
  after : List ComputeStmt

/-- A missing loop is explicit. Extraction never manufactures a loop or body. -/
def whileLoop? (code : List ComputeStmt) (occurrence : Nat := 0) : Option WhileLoop :=
  match code with
  | [] => none
  | st :: rest =>
      match st with
      | .whileLoop condition body =>
          match occurrence with
          | 0 => some ⟨[], condition, body, rest⟩
          | n+1 => (whileLoop? rest n).map fun loop =>
              { loop with before := st :: loop.before }
      | _ => (whileLoop? rest occurrence).map fun loop =>
          { loop with before := st :: loop.before }

end VeriTile.Triton.Source
