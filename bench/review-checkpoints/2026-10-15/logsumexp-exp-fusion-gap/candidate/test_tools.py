import unittest,sys,types,struct,collections,tempfile,hashlib,json
from pathlib import Path
from fractions import Fraction
from math import factorial
P=Path(__file__).resolve().parent
# Load exactly the audited Python bytes, not timestamp-sensitive bytecode.
def load(n):
 p=P/(n+'.py');m=types.ModuleType('lse_test_'+n);m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
m=load('inspect_lowering');c=load('capture');probe=load('probe')
def profile(scale,fusion):
 d=P/f'native/sm80-fp32-d17-b32-w4-scale{int(scale)}-fusion{int(fusion)}';return (d/'kernel.ptx').read_text(),(d/'kernel.ttir').read_text()
PTX,TTIR=profile(False,False)
class Checks(unittest.TestCase):
 def reject(self,p=PTX,t=TTIR,scale=False,fusion=False):
  with self.assertRaises(ValueError):m.inspect(p,t,scale,fusion)
 def inject(self,event,extra):
  lines=PTX.splitlines();lines.insert(event['line'],extra);return '\n'.join(lines)
 def chain(self):return m.inspect(PTX,TTIR,False,False)['ptx_exponent_chains'][0]
 def test_all_actual_profiles(self):
  r=json.loads((P/'native/receipt.json').read_text());self.assertEqual(r['compiled'],48)
  for row in r['records']:
   d=P/'native'/row['id'];m.inspect((d/'kernel.ptx').read_text(),(d/'kernel.ttir').read_text(),row['constexprs']['HAS_SCALE'],row['options']['enable_fp_fusion'])
 def test_wrong_log2e(self):self.reject(PTX.replace('0f3FB8AA3B','0f3F800000'))
 def test_exp_ftz(self):self.reject(PTX.replace('ex2.approx.f32','ex2.approx.ftz.f32'))
 def test_prefix_wrong(self):self.reject(PTX.replace('sub.rn.f32','add.rn.f32'))
 def test_prefix_overwrite(self):
  e=self.chain()['prefix'];self.reject(self.inject(e,f"mov.b32 {e['dst']}, 0;"))
 def test_mul_overwrite(self):
  e=self.chain()['log2e_multiply'];self.reject(self.inject(e,f"mov.b32 {e['dst']}, 0;"))
 def test_predicated_overwrite(self):
  e=self.chain()['log2e_multiply'];self.reject(self.inject(e,f"@%p999 mov.b32 {e['dst']}, 0;"))
 def test_multidst_overwrite(self):
  e=self.chain()['log2e_multiply'];self.reject(self.inject(e,f"ld.global.v2.b32 {{{e['dst']}, %r999}}, [%rd999];"))
 def test_unknown_possible_write(self):
  e=self.chain()['log2e_multiply'];self.reject(self.inject(e,f"unknown.op %r999, {e['dst']};"))
 def test_control_boundary(self):
  e=self.chain()['log2e_multiply'];self.reject(self.inject(e,'bra $L__BB999;\n$L__BB999: // actual branch target'))
 def test_predicated_prefix(self):self.reject(PTX.replace('sub.rn.f32','@%p999 sub.rn.f32'))
 def test_debug_named_control_target(self):
  e=self.chain()['log2e_multiply'];self.reject('bra $L__tmp999;\n'+self.inject(e,'$L__tmp999:'))
 def test_scaled_fusion_graph_is_distinct(self):
  p,t=profile(True,True);self.assertEqual(m.inspect(p,t,True,True)['ptx_exponent_chains'][0]['graph'],'SCALED_FMA');self.reject(p,t,True,False)
 def test_fma_negation_changed(self):
  p,t=profile(True,True);self.reject(p.replace('neg.f32','abs.f32'),t,True,True)
 def test_fma_wrong_operand(self):
  p,t=profile(True,True);e=m.inspect(p,t,True,True)['ptx_exponent_chains'][0]['prefix'];bad=e['text'].replace(e['args'][0],'%r999',1);self.reject(p.replace(e['text'],bad),t,True,True)
 def test_wrong_source_max_policy(self):self.reject(t=TTIR.replace('arith.maxnumf','arith.maximumf'))
 def test_wrong_source_exp_input(self):
  r=m.inspect(PTX,TTIR,False,False)['source_ssa'];self.reject(t=TTIR.replace('math.exp '+r['subtract'],'math.exp '+r['load']))
 def test_wrong_source_log_input(self):
  r=m.inspect(PTX,TTIR,False,False)['source_ssa'];self.reject(t=TTIR.replace('math.log '+r['sum'],'math.log '+r['maximum']))
 def test_wrong_padding(self):self.reject(t=TTIR.replace('0xFF800000','0x00000000'))
 def test_wrong_store_payload(self):
  r=m.inspect(PTX,TTIR,False,False)['source_ssa'];self.reject(t=TTIR.replace(', '+r['store_payload']+' : !tt.ptr<f32>',', '+r['log']+' : !tt.ptr<f32>'))
 def test_independent_exp_cells(self):
  for x,lo,hi in [(Fraction(-6291459,8388608),31699971,31699973),(Fraction(-12582919,16777216),31699969,31699971)]:
   total=sum(x**i/factorial(i) for i in range(16));err=abs(x)**16*Fraction(17,factorial(16)*16)
   self.assertGreater(total-err,Fraction(lo,67108864));self.assertLess(total+err,Fraction(hi,67108864))
 def test_fusion_exact_rational(self):
  self.assertEqual((1+Fraction(1,8388608))*(1-Fraction(1,8388608))-1,-Fraction(1,2**46))
 def test_scale_collision_cpu(self):
  f=lambda x:struct.unpack('<f',struct.pack('<I',x))[0];bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
  for x in [0xbf400006,0xbf400007]:self.assertEqual(bits(f(x)*f(0x3fb8aa3b)),0xbf8a7fb1)
 def test_fixture_complete_allocation_and_tail(self):
  for case in c.cases():
   a,b,active=c.fixture(case);self.assertEqual((len(a),len(b)),(67,19));self.assertTrue(all(8<=o<19 for o in active));self.assertEqual(a[8+2*17+1],case['input_word']);self.assertEqual(len(active),case['grid'][0]*case['grid'][1])
 def test_namedtuple_keeps_fields(self):
  M=collections.namedtuple('M','num_warps shared');self.assertEqual(probe.serial(M(4,64)),{'num_warps':4,'shared':64})
 def test_checked_source_rejects_drift(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.py';p.write_text('result=1\n');h=hashlib.sha256(p.read_bytes()).hexdigest();p.write_text('result=2\n')
   with self.assertRaises(ValueError):c.checked(p,'changed_source',h)
if __name__=='__main__':unittest.main()
