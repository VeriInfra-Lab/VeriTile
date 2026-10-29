NS='VeriTile.Bench.TritonBenchG.VectorAddition.'
def dtype(d):return {'float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1','float16':'fp16','float32':'fp32','float64':'fp64','bfloat16':'bf16','int1':'bool',
 'float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz',
 'fp8e4nv':'fp8e4m3fn','fp8e5':'fp8e5m2','f8e4':'fp8e4m3fn','f8e5':'fp8e5m2','float8e4nv':'fp8e4m3fn','float8e5':'fp8e5m2'}.get(d,d)

def callback(port):
 if port.startswith('b.rest.binary'):return 'VeriTile.Triton.FP.Structural.Algebra.binary'
 if port.startswith('b.rest.cast'):return 'VeriTile.Triton.FP.Structural.Algebra.cast'
 if port.startswith('b.wide.'):return NS+'FP64TypedCore.Numerics.'+port.split('.')[-1]
 if port.startswith('FP.'):return 'VeriTile.Triton.'+port
 return NS+port.split(' ')[0]

def operation_types(pair,p,op):
 port=op['port'];x,y=pair;compute=dtype(p['compute_dtype']);cfg=op['configuration']
 if port.startswith('b.rest.binary'):t=dtype(port.split()[2]);return [t,t],t
 if port.startswith('b.rest.cast'):a,b=port.split()[2:4];return [dtype(a)],dtype(b)
 if port in ['MixedSmall.integerRounded','MixedFP32.integerRounded']:
  integer=x if x=='bool' or x.startswith(('int','uint')) else y
  target='fp16' if '-> fp16' in op['operation'] else 'fp32'
  return ['uint1_after_bool_normalization' if integer=='bool' else integer],target
 if port=='MixedSmall.addTarget .fp16':return ['fp16','fp16'],'fp16'
 if port=='MixedSmall.widenHalf':return ['fp16'],'fp32'
 if port=='MixedSmall.widenBFloat':return ['bf16'],'fp32'
 if port=='MixedSmall.narrowBFloat':return ['fp32'],'bf16'
 if port=='FP.add':return ['fp32','fp32'],'fp32'
 if port.startswith('FP.IntegerConversion.'):
  return ['fp32'],('int' if port.endswith('toSigned') else 'uint')+str(cfg['width'])
 if port.startswith('b.wide.'):
  name=port.split('.')[-1]
  if name=='add64':return ['fp64','fp64'],'fp64'
  if name in ['signedTo64','unsignedTo64']:return [('int' if name=='signedTo64' else 'uint')+str(cfg['width'])],'fp64'
  if name in ['halfTo32','bfloatTo32']:return ['fp16' if name=='halfTo32' else 'bf16'],'fp32'
  if name=='singleTo64':return ['fp32'],'fp64'
  if name in ['e4To64','e5To64']:return ['fp8e4m3fn' if name=='e4To64' else 'fp8e5m2'],'fp64'
  if name in ['toSigned','toUnsigned']:return ['fp64'],('int' if name=='toSigned' else 'uint')+str(cfg['width'])
  return ['fp64'],{'toHalf':'fp16','toBFloat':'bf16','toSingle':'fp32','toE4RTNE':'fp8e4m3fn','toE5RTNE':'fp8e5m2'}[name]
 raise ValueError(port)
