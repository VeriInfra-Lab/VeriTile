import VeriTile.Triton.DSL
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SignedInteger16Cast
open VeriTile.Triton

def signed (x : Int) : ComputeKernel := triton {
  value = $((x : Int))
  narrowed = value.to(tl.int16)
  parenthesized = (value).to(tl.int16)
  functional = tl.cast(value, tl.int16)
}

noncomputable def results (k : ComputeKernel) (s : BlockState) :=
  (k.toAlgorithm?.toOption.bind (fun a => VeriTile.Triton.exec a s)).map
    (fun t => (t.regs .int [] "narrowed", t.regs .int [] "parenthesized", t.regs .int [] "functional"))

theorem signed_exec (x : Int) (s : BlockState) :
    results (signed x) s = some
      (some (Tile.scalar (BitVec.ofInt 16 x).toInt),
       some (Tile.scalar (BitVec.ofInt 16 x).toInt),
       some (Tile.scalar (BitVec.ofInt 16 x).toInt)) := by
  simp only [results, signed, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_intToSigned]
  simp [evalOp, BlockState.setReg, Tile.scalar, BitVec.toInt_ofNat', -Tile.scalar_eta]

theorem negative_one (s : BlockState) :
    results (signed (-1)) s = some
      (some (Tile.scalar (-1)), some (Tile.scalar (-1)), some (Tile.scalar (-1))) :=
  signed_exec (-1) s

theorem sign_boundary (s : BlockState) :
    results (signed (2^15)) s = some
      (some (Tile.scalar (-32768)), some (Tile.scalar (-32768)),
       some (Tile.scalar (-32768))) := signed_exec (2^15) s

theorem wrapped_positive (s : BlockState) :
    results (signed (2^16 + 7)) s = some
      (some (Tile.scalar 7), some (Tile.scalar 7), some (Tile.scalar 7)) :=
  signed_exec (2^16 + 7) s

theorem wrapped_negative (s : BlockState) :
    results (signed (-(2^16) - 3)) s = some
      (some (Tile.scalar (-3)), some (Tile.scalar (-3)), some (Tile.scalar (-3))) :=
  signed_exec (-(2^16) - 3) s

def loaded (Offsets : RegionName) : ComputeKernel := triton {
  tl.region Offsets = tl.int64
  direct = tl.load(Offsets).to(tl.int16)
  parens = (tl.load(Offsets)).to(tl.int16)
  deep = (((tl.load(Offsets)))).to(tl.int16)
  function = tl.cast(tl.load(Offsets), tl.int16)
}

def assignment (Offsets : RegionName) (name : RegName) : ComputeStmt :=
  .assign .int [] name (.alg (Op.intToSigned 16
    (.load .int (.region Offsets (.constNat 0)) .none .zero)))

theorem load_narrows_after_signed_read (Offsets : RegionName) :
    (loaded Offsets).surfaceBody = [assignment Offsets "direct", assignment Offsets "parens",
      assignment Offsets "deep", assignment Offsets "function"] := rfl

def maskCast : ComputeKernel := triton {
  offsets = tl.arange(0, 4)
  mask = offsets < $(2)
  values = mask.to(tl.int16)
  pos = tl.argmax(mask.to(tl.int16), 0)
}

theorem bool_before_argmax : maskCast.surfaceBody.getLast? = some
    (.assign .nat [] "pos" (.alg (.argMaxInt ⟨0, by simp⟩
      (Op.intToSigned 16 (Op.boolToInt (.ref .bool [4] "mask")))))) := rfl

#axiomsClean signedBits_int_mod
#axiomsClean eval_intToSigned
#axiomsClean evalOpR_intToSigned
#axiomsClean FP.Structural.eval_intToSigned
#axiomsClean eval_boolToInt
#axiomsClean evalOpR_boolToInt
#axiomsClean FP.Structural.eval_boolToInt

#guard_msgs (drop info) in
#auditModuleAxioms

def natural (n : Nat) : ComputeKernel := triton {
  value = $(n)
  narrowed = value.to(tl.int16)
  parenthesized = (value).to(tl.int16)
  functional = tl.cast(value, tl.int16)
}

theorem natural_exec (n : Nat) (s : BlockState) :
    results (natural n) s = some
      (some (Tile.scalar (BitVec.ofNat 16 n).toInt),
       some (Tile.scalar (BitVec.ofNat 16 n).toInt),
       some (Tile.scalar (BitVec.ofNat 16 n).toInt)) := by
  simp only [results, natural, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_intToSigned]
  simp [evalOp, BlockState.setReg, Tile.scalar, BitVec.toInt_ofNat', -Tile.scalar_eta]

theorem natural_sign_boundary (s : BlockState) :
    results (natural 32768) s = some
      (some (Tile.scalar (-32768)), some (Tile.scalar (-32768)),
       some (Tile.scalar (-32768))) := natural_exec 32768 s

/-- The scaled FP4 exponent has two explicit i16 narrowing events. -/
def fp4Scale (S : RegionName) : ComputeKernel := triton {
  tl.region S = tl.uint8
  s = tl.load(S)
  offset = (s.to(tl.int16) - $((127 : Int))).to(tl.int16)
  exponent = offset.to(tl.int32)
}

theorem fp4_scale_exact (S : RegionName) : (fp4Scale S).surfaceBody = [
    .assign .nat [] "s" (.alg (.load .nat (.region S (.constNat 0)) .none .zero)),
    .assign .int [] "offset" (.alg (Op.intToSigned 16
      (.sub .int .nil (Op.intToSigned 16 (.castNatToInt (.ref .nat [] "s")))
        (.constInt 127)))),
    .assign .int [] "exponent" (.alg (Op.intToSigned 32 (.ref .int [] "offset")))] := rfl

def unsignedLoaded (S : RegionName) : ComputeKernel := triton {
  tl.region S = tl.uint8
  a = tl.load(S).to(tl.int16)
  b = (((tl.load(S)))).to(tl.int16)
  c = tl.cast(tl.load(S), tl.int16)
}

theorem unsigned_load_before_signed_cast (S : RegionName) :
    (unsignedLoaded S).surfaceBody =
      ["a", "b", "c"].map (fun name => .assign .int [] name (.alg
        (Op.intToSigned 16 (.castNatToInt
          (.load .nat (.region S (.constNat 0)) .none .zero))))) := rfl

#axiomsClean signed_exec
#axiomsClean natural_exec
#axiomsClean sign_boundary
#axiomsClean natural_sign_boundary
#axiomsClean load_narrows_after_signed_read
#axiomsClean unsigned_load_before_signed_cast
#axiomsClean fp4_scale_exact
end VeriTile.Tests.SignedInteger16Cast
