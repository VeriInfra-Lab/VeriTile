import bench.repairs.tritonbench_g.block_sparse_attn.FPEquiv
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses
def zeroInstructions : Native.Instructions :=
  ⟨fun _ _=>0,fun _ _=>0,fun _ _=>0,fun _ _ _=>0,fun _ _=>0,fun _=>0,fun _ _=>0,fun _ _ _ _=>0⟩
def changedAdd : Native.Instructions := {zeroInstructions with add:=fun _ _=>1}
theorem numeric_not_automatic : ¬Native.Correspondence zeroInstructions changedAdd := by
  intro h
  have he:=h.add 0 0
  have bad : (0 : Word)=1 := he
  contradiction

def shape64 : Shape := ⟨2,4,2,32,16,16,16,4,1⟩
theorem full_dimension : shape64.dim=64 := rfl
theorem shape_valid : shape64.Valid := by unfold Shape.Valid shape64;decide
theorem last_channel_covered :
    laneWork shape64 ⟨1,7,3,15,15⟩=⟨1,3,31,63⟩ := by decide
theorem last_channel_active : (Work.mk 1 3 31 63).active shape64 := by decide
theorem final_output_address : (Work.mk 1 3 31 63).target shape64=16383 := by decide
theorem stride_two_channel31 : address ⟨4096,2048,64,2⟩ 0 0 0 31=62 := rfl
theorem fourth_block_value : Source.valuePointer ⟨8192,4096,64,1⟩ 0 0 0 16 0 3 16 0=48 := by decide
theorem codec_nan_payload : Native.load (fun _ i=>Native.encode (0x7fc01234 : Word) ⟨i%4,by omega⟩) .q 0=
    (0x7fc01234 : Word) := by decide

#axiomsClean numeric_not_automatic
#axiomsClean shape_valid
#axiomsClean last_channel_covered
#axiomsClean last_channel_active
#axiomsClean final_output_address
#axiomsClean stride_two_channel31
#axiomsClean fourth_block_value
#axiomsClean codec_nan_payload
end VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses
