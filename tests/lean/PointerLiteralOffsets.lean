import VeriTile.Triton.DSL.PointerLiteralOffsets
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PointerLiteralOffsets
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 1500000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("rows",.i32)]
  scalars := [("base",.integer)]
  readPorts := ["rows"]
def source (rows : Region .int) (base : Nat) : ComputeKernel :=
  with_triton_call signature call_args [base := base] in triton {
    ptr = rows + base
    first = tl.load(ptr + 1).to(tl.int32)
    second = tl.load(ptr + 2).to(tl.int32)
  }

/-- Both distinct offsets preserve the complete typed load program for every
caller region/base; neither literal becomes a floating operation. -/
theorem complete_program (rows : Region .int) (base : Nat) : source rows base =
    .mk [Region.cast rows] []
      [.assign .ptr [] "ptr" (.alg (.ptrAdd .nil (.ptrBase rows) (.constNat base))),
       .assign .int [] "first" (.alg (.load .int (.ptr (.ptrAdd .nil (.ref .ptr [] "ptr") (.constNat 1))) .none)),
       .assign .int [] "second" (.alg (.load .int (.ptr (.ptrAdd .nil (.ref .ptr [] "ptr") (.constNat 2))) .none))] := rfl

def initial : BlockState :=
  {mem := fun _ o => MemCell.of .int ((o : Int)-10),
   regs := fun _ _ _ => none, pids := fun _ => 0}
def result (rows : Region .int) (base : Nat) : BlockState :=
  let s := initial.setReg "ptr" .ptr [] (Tile.scalar (Region.cast rows,base))
  let s := s.setReg "first" .int [] (Tile.scalar (((base+1 : Nat) : Int)-10))
  s.setReg "second" .int [] (Tile.scalar (((base+2 : Nat) : Int)-10))

/-- Execute both actual loads against address-dependent, signed memory. This
would detect dropping an offset or reading the same CSR entry twice. -/
theorem actual_integer_reads (rows : Region .int) (base : Nat) :
    stepStmts (source rows base).toAlgKernel.body initial=some (result rows base) := by
  simp [complete_program,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,bind,Option.bind,Except.bind,
    initial,result,BlockState.readMemValue,BlockState.readMemTyped,MemCell.readAs,MemCell.of,
    NumericDType.add,Tile.bop,Tile.scalar,Tile.ptrAdd,Broadcast.leftIndex,Broadcast.rightIndex,
    Region.cast,-Tile.scalar_eta]

def rebound (rows : Region .int) (base : Nat) : ComputeKernel :=
  with_triton_call signature call_args [base := base] in triton {
    ptr = rows + base
    ptr = 2.5
    value = ptr + 1
  }
/-- Rebinding clears the pointer fact: ordinary floating addition keeps its
literal precision instead of inheriting an address carrier. -/
theorem rebound_is_floating (rows : Region .int) (base : Nat) :
    (rebound rows base).surfaceBody[2]?=some
      (.assign .real [] "value" (.alg
        (.add .real .nil (.ref .real [] "ptr") (.const 1)))) := rfl

def localLoop (rows : Region .int) (base : Nat) : ComputeKernel :=
  with_triton_call signature call_args [base := base] in triton {
    for i in range(0, 2):
        ptr = rows + base
        value = tl.load(ptr + 1).to(tl.int32)
  }
/-- A loop-local pointer receives its fact at its defining statement; its
load stays inside the loop and uses the same arbitrary region and base. -/
theorem loop_scope (rows : Region .int) (base : Nat) : localLoop rows base=
    .mk [Region.cast rows] []
      [.forRange "i" 0 2 1
        [.assign .ptr [] "ptr" (.alg (.ptrAdd .nil (.ptrBase rows) (.constNat base))),
         .assign .int [] "value" (.alg (.load .int
           (.ptr (.ptrAdd .nil (.ref .ptr [] "ptr") (.constNat 1))) .none))]] := rfl

#axiomsClean complete_program
#axiomsClean actual_integer_reads
#axiomsClean rebound_is_floating
#axiomsClean loop_scope
end
end VeriTile.Tests.PointerLiteralOffsets
