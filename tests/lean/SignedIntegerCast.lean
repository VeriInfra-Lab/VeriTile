import VeriTile.Triton.DSL
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SignedIntegerCast
open VeriTile.Triton

def signed (x : Int) : ComputeKernel := triton {
  value = $((x : Int))
  narrowed = value.to(tl.int32)
  parenthesized = (value).to(tl.int32)
  functional = tl.cast(value, tl.int32)
}

noncomputable def results (k : ComputeKernel) (s : BlockState) :=
  (k.toAlgorithm?.toOption.bind (fun a => VeriTile.Triton.exec a s)).map
    (fun t => (t.regs .int [] "narrowed", t.regs .int [] "parenthesized", t.regs .int [] "functional"))

theorem signed_exec (x : Int) (s : BlockState) :
    results (signed x) s = some
      (some (Tile.scalar (BitVec.ofInt 32 x).toInt),
       some (Tile.scalar (BitVec.ofInt 32 x).toInt),
       some (Tile.scalar (BitVec.ofInt 32 x).toInt)) := by
  simp only [results, signed, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_intToSigned]
  simp [evalOp, BlockState.setReg, Tile.scalar, -Tile.scalar_eta]

theorem negative_one (s : BlockState) :
    results (signed (-1)) s = some
      (some (Tile.scalar (-1)), some (Tile.scalar (-1)), some (Tile.scalar (-1))) :=
  signed_exec (-1) s

theorem sign_boundary (s : BlockState) :
    results (signed (2^31)) s = some
      (some (Tile.scalar (-2147483648)), some (Tile.scalar (-2147483648)),
       some (Tile.scalar (-2147483648))) := signed_exec (2^31) s

theorem wrapped_positive (s : BlockState) :
    results (signed (2^32 + 7)) s = some
      (some (Tile.scalar 7), some (Tile.scalar 7), some (Tile.scalar 7)) :=
  signed_exec (2^32 + 7) s

theorem wrapped_negative (s : BlockState) :
    results (signed (-(2^32) - 3)) s = some
      (some (Tile.scalar (-3)), some (Tile.scalar (-3)), some (Tile.scalar (-3))) :=
  signed_exec (-(2^32) - 3) s

def loaded (Offsets : RegionName) : ComputeKernel := triton {
  tl.region Offsets = tl.int64
  direct = tl.load(Offsets).to(tl.int32)
  parens = (tl.load(Offsets)).to(tl.int32)
  deep = (((tl.load(Offsets)))).to(tl.int32)
  function = tl.cast(tl.load(Offsets), tl.int32)
}

def assignment (Offsets : RegionName) (name : RegName) : ComputeStmt :=
  .assign .int [] name (.alg (Op.intToSigned 32
    (.load .int (.region Offsets (.constNat 0)) .none .zero)))

theorem load_narrows_after_signed_read (Offsets : RegionName) :
    (loaded Offsets).surfaceBody = [assignment Offsets "direct", assignment Offsets "parens",
      assignment Offsets "deep", assignment Offsets "function"] := rfl

def maskCast : ComputeKernel := triton {
  offsets = tl.arange(0, 4)
  mask = offsets < $(2)
  values = mask.to(tl.int32)
  pos = tl.argmax(mask.to(tl.int32), 0)
}

theorem bool_before_argmax : maskCast.surfaceBody.getLast? = some
    (.assign .nat [] "pos" (.alg (.argMaxInt ⟨0, by simp⟩
      (Op.boolToInt (.ref .bool [4] "mask"))))) := rfl

#axiomsClean signedBits_int_mod
#axiomsClean eval_intToSigned
#axiomsClean evalOpR_intToSigned
#axiomsClean FP.Structural.eval_intToSigned
#axiomsClean eval_boolToInt
#axiomsClean evalOpR_boolToInt
#axiomsClean FP.Structural.eval_boolToInt

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.SignedIntegerCast
