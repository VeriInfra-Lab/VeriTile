import bench.repairs.tritonbench_g.flash_attn.FPEquiv
import bench.repairs.tritonbench_g.flash_attn.Proofs.Bounds
-- Preserve the historical Bool spellings after importing the shared Triton DSL.
local notation "true" => Bool.true
local notation "false" => Bool.false
namespace VeriTile.Bench.Repairs.FlashAttn.Witnesses

def zeroNative : Native.Instructions :=
  ⟨fun _ _=>0,fun _ _=>0,fun _ _=>0,fun _ _ _=>0,fun _ _=>0,fun _=>0,fun _=>0,
    fun _ _=>0,fun _=>0,fun _=>0,fun _ _ _ _=>0⟩
def changedNative : Native.Instructions := {zeroNative with mul:=fun _ _=>1}

theorem numeric_not_automatic : ¬Native.Correspondence zeroNative changedNative := by
  intro h
  have hh:=h.mul 0 0
  change (0 : Word)=1 at hh
  exact (by decide : (0 : Word)≠1) hh

theorem native_reflexive : Native.Correspondence zeroNative zeroNative := Native.reflexive _

theorem raw_nan_codec : Physical.decode (Physical.encode (0x7fc01234 : Word))=0x7fc01234 :=
  Physical.decode_encode _

theorem original_v_head_address :
    1*16384+0*64+0=16384 ∧ address ⟨65536,32768,64,1⟩ 0 1 0 0=32768 := by decide

theorem tail_65 (causal : Bool) :
    rounds ⟨1,1,65,16,causal⟩ 0=2 ∧
    visible ⟨1,1,65,16,causal⟩ 64 64=true ∧
    visible ⟨1,1,65,16,causal⟩ 64 65=false := by cases causal <;> decide

theorem padded_row_excluded (causal : Bool) :
    ¬(Work.output 0 0 65 0).active ⟨1,1,65,16,causal⟩ := by simp [Work.active]
#axiomsClean numeric_not_automatic
#axiomsClean native_reflexive
#axiomsClean raw_nan_codec
#axiomsClean original_v_head_address
#axiomsClean tail_65
#axiomsClean padded_row_excluded
end VeriTile.Bench.Repairs.FlashAttn.Witnesses
