import bench.repairs.tritonbench_g.int4_matmul.Misc.RawNative
import bench.repairs.tritonbench_g.int4_matmul.Misc.HistoricalFPEquiv
namespace VeriTile.Bench.Repairs.Int4Matmul.Witnesses
open Helpers Physical

def tiny : Shape := ⟨3,8,8,8⟩
theorem tiny_admitted : tiny.Valid := by constructor <;> decide +kernel
theorem original_unmasked_tail : ¬(kIndex ⟨32,16,512,16,2⟩ 1 0 511<tiny.k) := by decide +kernel
theorem tail_address : kIndex ⟨32,16,512,16,2⟩ 1 0 511=1023 := by decide +kernel
theorem fresh_seed_defect : (1:Word)+packLoop (fun _ => (0:BitVec 4)) 8≠packedSpec (fun _ => (0:BitVec 4)) := by decide +kernel
theorem signed_high_nibble : nibble (0xffffffff:Word) 7=15 := by decide +kernel
theorem unpack_transpose_distinct : 3*16+7=55 ∧ 7*8+3=59 ∧ 55≠59 := by decide +kernel
theorem opaque_nan_roundtrip : decode (encode (0x7fc01234:Word))=(0x7fc01234:Word) := decode_encode _
theorem invalid_group_rejected : ¬(Shape.Valid ⟨1,8,8,0⟩) := by
  intro h
  have := h.gpos
  change 0<0 at this
  omega

def zeroMatrix : Native.Instructions := ⟨(fun _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ _ _ => 0)⟩
def badMatrix : Native.Instructions := {zeroMatrix with convert := fun _ => 1}
theorem matrix_numeric_not_automatic : ¬Native.Correspondence zeroMatrix badMatrix := by
  intro h
  have he := h.convert 0
  change (0:Word)=1 at he
  exact (by decide : (0:Word)≠1) he

def zeroHelper : HelperNative.Instructions :=
  ⟨(fun _ => 0),(fun _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ => 0),
   (fun _ _ => 0),(fun _ _ => 0),(fun _ => 0),(fun _ => 0),(fun _ => 0),(fun _ => 0),(fun _ => 0)⟩
def badHelper : HelperNative.Instructions := {zeroHelper with narrow := fun _ => 1}
theorem helper_numeric_not_automatic : ¬HelperNative.Correspondence zeroHelper badHelper := by
  intro h
  have he := h.narrow 0
  change (0:BitVec 16)=1 at he
  exact (by decide : (0:BitVec 16)≠1) he

theorem native_consistency : WholeNative zeroMatrix zeroMatrix zeroHelper zeroHelper := whole_native _ _
#axiomsClean tiny_admitted
#axiomsClean original_unmasked_tail
#axiomsClean tail_address
#axiomsClean fresh_seed_defect
#axiomsClean signed_high_nibble
#axiomsClean unpack_transpose_distinct
#axiomsClean opaque_nan_roundtrip
#axiomsClean invalid_group_rejected
#axiomsClean matrix_numeric_not_automatic
#axiomsClean helper_numeric_not_automatic
#axiomsClean native_consistency
end VeriTile.Bench.Repairs.Int4Matmul.Witnesses
