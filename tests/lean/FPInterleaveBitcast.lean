import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FPInterleaveBitcast
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 400000
set_option linter.unusedSimpArgs false

def source (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.int32, Y = tl.float32
  lane = tl.arange(0, 2)
  a = tl.load(X + lane)
  b = tl.load(X + lane + $(2))
  ordered = tl.interleave(a, b)
  value = ordered.to(tl.float32, bitcast=True)
  tl.store(Y + tl.arange(0, 4), value)
}

theorem source_interleave : (source "X" "Y").surfaceBody[3]? = some
    (.assign .int [2 * 2] "ordered" (.alg (.reshape [2 * 2]
      (.join (.ref .int [2] "a") (.ref .int [2] "b"))))) := rfl

theorem source_bitcast : (source "X" "Y").surfaceBody[4]? = some
    (.assign .real [4] "value" (.compute
      (.bitcast .int32 .fp32 rfl (.alg .int32 (.ref .int [4] "ordered"))))) := rfl

/-- Reinterpretation is parameterized only by the backend's representation of
the exact 32 payload bits. It is not the backend's integer-to-float conversion. -/
theorem signed_bits {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (shape : TileShape) (v : Values α .int shape) :
    evalComputeOp M (.bitcast .int32 .fp32 rfl
      (.alg .int32 (.ref .int shape "raw"))) (s.setReg "raw" .int shape v) =
    some (fun i => M.fp32Bits ⟨BitVec.ofInt 32 (v i)⟩) := by
  simp [evalComputeOp, evalOp_unfold, State.setReg, ComputeDType.eraseDType]

theorem unsigned_bits {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (shape : TileShape) (v : Values α .nat shape) :
    evalComputeOp M (.bitcast .uint32 .fp32 rfl
      (.alg .uint32 (.ref .nat shape "raw"))) (s.setReg "raw" .nat shape v) =
    some (fun i => M.fp32Bits ⟨BitVec.ofNat 32 (v i)⟩) := by
  simp [evalComputeOp, evalOp_unfold, State.setReg, ComputeDType.eraseDType]

theorem split_append (shape : TileShape) (i : TileIndex shape) (side : Fin n) :
    TileShape.splitLastIndex shape (TileShape.appendAxisIndex shape i side) = (i, side) := by
  induction shape with
  | nil => cases i; rfl
  | cons d tail ih =>
    rcases i with ⟨i,j⟩
    simp [TileShape.splitLastIndex, TileShape.appendAxisIndex, ih]

/-- Join and split rearrange arbitrary typed values; they never invoke a
floating arithmetic callback or change the current compute precision. -/
theorem split_join {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (a b : Op dtype shape)
    (va vb : Values α dtype shape) (ha : evalOp M p a s = some va)
    (hb : evalOp M p b s = some vb) (side : Fin 2) :
    evalOp M p (.split side (.join a b)) s =
      some (fun i => if side.val=0 then va i else vb i) := by
  change ((evalOp M p a s).bind fun av => (evalOp M p b s).bind fun bv =>
    some (fun i : TileIndex (shape ++ [2]) =>
      if (TileShape.splitLastIndex shape i).2.val = 0 then
        av (TileShape.splitLastIndex shape i).1 else bv (TileShape.splitLastIndex shape i).1)).bind
      (fun v => some (fun i => v (TileShape.appendAxisIndex shape i side))) = _
  rw [ha, hb]
  simp only [Option.bind_some]
  apply congrArg some
  funext i
  rw [split_append]

def scalarInterleave : ComputeKernel := triton {
  pair = tl.interleave($(1), $(2))
}

theorem scalar_shape : scalarInterleave.surfaceBody =
    [.assign .nat [2] "pair" (.alg (.join (.constNat 1) (.constNat 2)))] := rfl

def matrixInterleave : ComputeKernel := triton {
  a = tl.full([2, 3], $(1))
  b = tl.full([2, 3], $(2))
  pair = tl.interleave(a, b)
}

theorem matrix_last_axis : matrixInterleave.surfaceBody[2]? = some
    (.assign .real [2,6] "pair" (.alg (.reshape [2,6]
      (.join (.ref .real [2,3] "a") (.ref .real [2,3] "b"))))) := rfl

def integerBits : ComputeKernel := triton {
  u = $((2147483648 : Nat))
  s = u.to(tl.int32)
  joined = s | 3
  masked = joined & 7
}

theorem unsigned_to_signed_width : integerBits.surfaceBody[1]? = some
    (.assign .int [] "s" (.alg (.intToSigned 32 (.castNatToInt (.ref .nat [] "u"))))) := rfl

/-- The trace backend keeps payloads as Nat and deliberately makes numeric
integer conversion different, so using that conversion would fail the test. -/
private noncomputable def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ v => v
  fromNat := fun _ _ => 123
  fromInt := fun _ _ => 456
  fp32Bits := fun v => v.bits.toNat
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

private def initial : State Nat where
  mem := fun _ i => .mk .int (match i with
    | 0 => 0x3f800000 | 1 => -2147483648 | 2 => 0x3f000000 | _ => 0x7fc01234)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Full load/interleave/runtime-bitcast/store execution. Includes -0 and a
NaN payload as raw bits and checks the unchanged source memory. -/
theorem ordered_payload_execution :
    ∃ t, run model (source "X" "Y").surfaceBody initial = some t ∧
      t.mem "Y" 0 = .mk .real 0x3f800000 ∧
      t.mem "Y" 1 = .mk .real 0x3f000000 ∧
      t.mem "Y" 2 = .mk .real 0x80000000 ∧
      t.mem "Y" 3 = .mk .real 0x7fc01234 ∧
      t.mem "X" 1 = .mk .int (-2147483648) := by
  refine ⟨?t, ?execution, ?outputs⟩
  case execution =>
    simp [source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, model, initial, State.setReg, store, natLt]
    rfl
  case outputs => exact ⟨rfl, rfl, rfl, rfl, rfl⟩

#axiomsClean source_interleave
#axiomsClean source_bitcast
#axiomsClean signed_bits
#axiomsClean unsigned_bits
#axiomsClean split_append
#axiomsClean split_join
#axiomsClean scalar_shape
#axiomsClean matrix_last_axis
#axiomsClean unsigned_to_signed_width
#axiomsClean ordered_payload_execution
end VeriTile.Tests.FPInterleaveBitcast
