import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete typed source pair, under development before public migration.
The candidate reverses the operands of the discrete comparison defining the
upper triangular matrix. Every loop, load, cast, store and carry update stays
in source order. In particular, block loads retain unspecified padding.

The signed mirrored range preserves the negative start and stop, so T = 0
executes no loop iterations. Dynamic signed block offsets are projected to
Nat only after the complete signed multiplication. Source storage loads and
pointer-element output casts remain explicit in every dtype profile.
-/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedSource
open Lean VeriTile.Triton

syntax "reversedVectorSource%" ident : term
set_option hygiene false in
macro_rules
  | `(reversedVectorSource% $version:ident) => do
    unless version.getId == `original || version.getId == `compareRight do
      Macro.throwError "expected original or compareRight"
    let triangular : TSyntax `tritonStmt ←
      if version.getId == `compareRight then
        `(tritonStmt| m_s = tl.where(o_i[None, :] >= o_i[:, None], 1.0, 0.0))
      else
        `(tritonStmt| m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0))
    `(fun (dtype out_dtype : FloatDType) (s z : RegionName)
        (s_s_h s_s_t s_s_d T S BT BS : Nat) =>
      ComputeKernel.mk [s] [z] (triton {
        tl.region s = DTYPE, z = OUT_DTYPE
        i_s = tl.program_id(0)
        i_bh = tl.program_id(1)
        o_i = tl.arange(0, $(BT))
        $triangular
        b_z = tl.zeros([$(BS)], dtype=tl.float32)
        for i_t in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
          p_s = tl.make_block_ptr(base=s + i_bh * $(s_s_h),
            shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
            offsets=(i_t * $(BT), i_s * $(BS)),
            block_shape=($(BT), $(BS)), order=(1, 0))
          p_z = tl.make_block_ptr(base=z + i_bh * $(s_s_h),
            shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
            offsets=(i_t * $(BT), i_s * $(BS)),
            block_shape=($(BT), $(BS)), order=(1, 0))
          b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
          b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
          tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
          if i_t >= $(0) {
            b_z += tl.sum(b_s, 0)
          }
        }
      }).surfaceBody)

def original := reversedVectorSource% original
def compareRight := reversedVectorSource% compareRight
def source (swapped : Bool) := if swapped then compareRight else original

end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedSource
