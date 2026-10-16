import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Staged full-source port. The complete original reverse loop, unpadded
boundary load, carry update, forward scan and typed store remain present.
The candidate changes only the order of one integer row-stride product in
the input block pointer. This module makes no completed correctness claim.
-/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedSource
open Lean VeriTile.Triton

syntax "reversedScalarSource%" ident : term
set_option hygiene false in
macro_rules
  | `(reversedScalarSource% $version:ident) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let ptr : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| p_s = tl.make_block_ptr(base=s+$(T)*i_bh,
          shape=($(T)), strides=($(1)), offsets=(i_t*$(BT)),
          block_shape=($(BT)), order=(0)))
      else
        `(tritonStmt| p_s = tl.make_block_ptr(base=s+i_bh*$(T),
          shape=($(T)), strides=($(1)), offsets=(i_t*$(BT)),
          block_shape=($(BT)), order=(0)))
    `(fun (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) =>
      ComputeKernel.mk [s] [o] (triton {
        tl.region s = DTYPE, o = OUT_DTYPE
        i_bh = tl.program_id(0)
        b_z = tl.zeros([], dtype=tl.float32)
        for i_t in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
          $ptr
          p_o = tl.make_block_ptr(base=o+i_bh*$(T), shape=($(T)), strides=($(1)),
            offsets=(i_t*$(BT)), block_shape=($(BT)), order=(0))
          b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
          b_zz = tl.sum(b_s, axis=0)
          b_z += b_zz
          b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
          tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0] : List Nat))
        }
      }).surfaceBody)

def original := reversedScalarSource% original
def indexCommuted := reversedScalarSource% indexCommuted

def source (swapped : Bool) := if swapped then indexCommuted else original

theorem projects (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (source swapped dtype out_dtype s o T BT).toAlgorithm? =
      .ok (source swapped dtype out_dtype s o T BT).toAlgKernel := by
  cases swapped <;> rfl

theorem statement_count (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (source swapped dtype out_dtype s o T BT).surfaceBody.length = 3 := by
  cases swapped <;> rfl

#axiomsClean projects
#axiomsClean statement_count

end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedSource
