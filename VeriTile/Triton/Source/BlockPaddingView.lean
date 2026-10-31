import VeriTile.Triton.Core.Ast
import VeriTile.Meta.StatementAudit

/-! Proof-only block-load padding view. Source kernels are never constructed
with this helper. A caller must prove the complete source connection and the
semantic padding condition before transferring an execution theorem. Ordinary
pointer loads, all types, numerical operation order and effects are retained. -/
namespace VeriTile.Triton.BlockPaddingView

def operation (padding : PaddingOption) : {d : TileDType} → {sh : TileShape} → Op d sh → Op d sh
  | _,_, .load d (.blockPtr p axes) mask _ => .load d (.blockPtr p axes) mask padding
  | _,_, .add d bc a b => .add d bc (operation padding a) (operation padding b)
  | _,_, .castFloat src dst a => .castFloat src dst (operation padding a)
  | _,_, .withPrecision d p a => .withPrecision d p (operation padding a)
  | _,_, x => x

def compute (padding : PaddingOption) : {d : ComputeDType} → {sh : TileShape} → ComputeOp d sh → ComputeOp d sh
  | _,_, .alg d a => .alg d (operation padding a)
  | _,_, .load d (.blockPtr p axes) mask _ => .load d (.blockPtr p axes) mask padding
  | _,_, x => x

def expression (padding : PaddingOption) : {d : AlgDType} → {sh : TileShape} → ComputeExpr d sh → ComputeExpr d sh
  | _,_, .alg a => .alg (operation padding a)
  | _,_, .compute a => .compute (compute padding a)
  | _,_, x => x

mutual
def algorithmStatement (padding : PaddingOption) : Stmt → Stmt
  | .assign d sh n e => .assign d sh n (operation padding e)
  | .forLoop i n body => .forLoop i n (algorithmStatements padding body)
  | .forRange i a b c body => .forRange i a b c (algorithmStatements padding body)
  | .forRangeDyn i a b c body => .forRangeDyn i a b c (algorithmStatements padding body)
  | .forRangeInt i a b c body => .forRangeInt i a b c (algorithmStatements padding body)
  | .ifThen cond body => .ifThen cond (algorithmStatements padding body)
  | .ifThenElse cond yes no => .ifThenElse cond (algorithmStatements padding yes) (algorithmStatements padding no)
  | x => x
termination_by x => sizeOf x
def algorithmStatements (padding : PaddingOption) : List Stmt → List Stmt
  | [] => []
  | x::xs => algorithmStatement padding x :: algorithmStatements padding xs
termination_by xs => sizeOf xs
end

mutual
def statement (padding : PaddingOption) : ComputeStmt → ComputeStmt
  | .alg st => .alg (algorithmStatement padding st)
  | .assign d sh n e => .assign d sh n (expression padding e)
  | .forLoop i n body => .forLoop i n (statements padding body)
  | .forRange i a b c body => .forRange i a b c (statements padding body)
  | .forRangeDyn i a b c body => .forRangeDyn i a b c (statements padding body)
  | .forRangeInt i a b c body => .forRangeInt i a b c (statements padding body)
  | .ifThen cond body => .ifThen cond (statements padding body)
  | .ifThenElse cond yes no => .ifThenElse cond (statements padding yes) (statements padding no)
  | x => x
termination_by x => sizeOf x
def statements (padding : PaddingOption) : List ComputeStmt → List ComputeStmt
  | [] => []
  | x::xs => statement padding x :: statements padding xs
termination_by xs => sizeOf xs
end

def kernel (padding : PaddingOption) : ComputeKernel → ComputeKernel
  | .mk ins outs body => .mk ins outs (statements padding body)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.BlockPaddingView
