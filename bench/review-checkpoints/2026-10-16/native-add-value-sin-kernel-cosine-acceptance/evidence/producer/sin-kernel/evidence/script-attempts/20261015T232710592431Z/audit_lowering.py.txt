from pathlib import Path
import hashlib,json,re
W=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();nr=json.loads((W/'evidence/native-latest.json').read_text());n=W/nr['receipt'];checks={}
for d,bits in [('float32',32),('float64',64)]:
 raw=(n.parent/(d+'.frontend.ttir')).read_text();ll=(n.parent/(d+'.llir')).read_text();ptx=(n.parent/(d+'.ptx')).read_text()
 assert raw.count('tt.load')==1 and raw.count('tt.store')==1 and raw.index('tt.load')<raw.index('math.sin')<raw.index('tt.store')
 assert 'tensor<1024xf'+str(bits)+'>' in raw and 'arith.cmpi slt' in raw and 'tt.assert' not in raw
 assert 'shl i32' in ll and 'icmp slt i32' in ll and 'sext i32' in ll and 'getelementptr' in ll
 assert 'shl.b32' in ptx and 'setp.lt.s32' in ptx
 checks[d]={'one_masked_load_sine_store_in_order':True,'no_runtime_source_assert':True,'same_width_sine_and_storage':bits,'backend_signed32_mask_and_pointer_sign_extension':True,'raw_sha256':sha(n.parent/(d+'.frontend.ttir')),'llir_sha256':sha(n.parent/(d+'.llir')),'ptx_sha256':sha(n.parent/(d+'.ptx'))}
abi=sorted((W/'evidence/abi-profiles').glob('*/receipt.json'));assert len(abi)==1;ar=json.loads(abi[0].read_text());assert ar['profiles']==30 and not ar['gpu_executed']
for p,h in ar['artifacts'].items():assert sha(abi[0].parent/p)==h
result={'status':'passed','native_receipt':nr,'source_instructions_and_lowering':checks,'original_binder_profiles':{'path':str(abi[0].relative_to(W)),'sha256':sha(abi[0]),'profiles':30},'source_code_refs':['SinKernelNative.'+n for n in ['original_ast_lowering','source_active_tile','source_pointer','compiled_scalar','retained_source_precision']],'numerics_validated':False,'GPU_launches':0}
(W/'evidence/source-lowering-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
