import VeriTile.Triton.DSL.PastedControlSource
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PastedControlSource
open VeriTile.Triton DSL.PastedControlSource ControlSource

def signature : CallSignature where
  pointers := [("X",.fp32)]
  scalars := [("k",.integer)]
  constexpr := ["NEG"]

def sample : Invocation :=
  with_control_call signature constexpr_bools [NEG := false] in triton {
    value = tl.program_id(0).to(tl.int64)
    ready = -1 if NEG else 1
    if NEG:
        value = value + k
    while value < k:
        pass
  }

/-- The following line's if is a new statement, not part of a ternary. -/
theorem separate_statements : sample.body.length=4 := rfl

/-- A false caller flag remains a branch in the source AST. A typed backend,
not the parser, must justify any later specialization. -/
theorem branch_retained :
    (match sample.body[2]? with
    | some (VeriTile.Triton.ControlSource.Statement.branch _ yes no) => yes.length=1 ∧ no=[]
    | _ => False) := by exact ⟨rfl,rfl⟩

theorem while_retained :
    (match sample.body[3]? with
    | some (VeriTile.Triton.ControlSource.Statement.whileLoop _ [VeriTile.Triton.ControlSource.Statement.pass]) => True
    | _ => False) := trivial

#axiomsClean separate_statements
#axiomsClean branch_retained
#axiomsClean while_retained
end VeriTile.Tests.PastedControlSource
