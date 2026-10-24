import json
from pathlib import Path
P=Path(__file__).resolve().parent
PROFILES=('source_separate_dot_add','compiled_separate_ascending_fma','compiled_fused_ascending_fma')
ORIGIN=8

def build_case(mode):
 arrays={
  'a_ptr':(16,4096,[0x3c00]*(4096+16),'A'),
  'b_ptr':(32,1024,[0x33333333]*(1024+16),'B'),
  'c_ptr':(16,8192,[0x7e55]*(8192+16),'C'),
  'scales_ptr':(16,256,[0x3c01]*8+[0x3c01]*128+[0x3c00]*(128+8),'S'),
  'zeros_ptr':(32,32,[0x22222222]*(32+16),'Z')}
 alloc={name:{'dtype':'uint16_storage_fp16' if width==16 else 'int32_packed','word_bits':width,
    'logical_origin_elements':ORIGIN,'logical_capacity_elements':size,'region':region,'raw_words':words}
  for name,(width,size,words,region) in arrays.items()}
 constants={'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8,'NO_GROUPS':mode}
 args={'M':64,'N':128,'K':64,'stride_am':64,'stride_ak':1,'stride_bk':128,'stride_bn':1,
  'stride_cm':128,'stride_cn':1,'stride_scales_g':128,'stride_scales_n':1,'stride_zeros_g':16,'stride_zeros_n':1,
  'groupsize':64 if mode else 32}
 bits={name:(0x5401 if mode else 0x5400) if name=='compiled_fused_ascending_fma' else (0x5402 if mode else 0x5401) for name in PROFILES}
 expected={name:[0x7e55]*8+[value]*8192+[0x7e55]*8 for name,value in bits.items()}
 return {'case_id':'nogroups_'+str(mode).lower(),'runtime_args':args,
  'launch':{'grid':[1,1,1],'constexprs':constants,'num_warps':4,'num_stages':4},
  'allocations':alloc,'expected_full_c_allocations_by_software_graph':expected,
  'expected_active_words':bits,'active_C_word_count':8192,
  'scope':'Dense legal 64x128x32 original autotune member, M64/N128/K64, all 8192 outputs active, both complete K blocks; S/Z retain two-group allocation capacity even when NO_GROUPS uses only the first group.'}

def make():
 return {'schema':1,'task':'matmul_dequantize_int4','source_sha256':'163c65f3c3c3c82cd0df16dd558662a8c9161d7b2af5ce9dca672384ffd140ed',
  'profiles':list(PROFILES),'cases':{x['case_id']:x for x in [build_case(False),build_case(True)]},
  'lean_namespace':'VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization',
  'lean_value_theorems':['original_output_all_lanes','separate_output_all_lanes','fused_output_all_lanes'],
  'lean_full_execution_theorem':'complete_legal_tile_execution','lean_physical_bridge':'physical_reference_bridge',
  'lean_bounds_theorems':['a_read_extent','b_read_extent','scale_read_extent','zero_read_extent','original_launch_geometry'],
  'native_conformance_proved':False,'discriminates':'half scalar fusion only; original and unfused live-carry graph happen to agree on this fixed case'}
if __name__=='__main__':(P/'experiment-case.json').write_text(json.dumps(make(),indent=2)+'\n')
