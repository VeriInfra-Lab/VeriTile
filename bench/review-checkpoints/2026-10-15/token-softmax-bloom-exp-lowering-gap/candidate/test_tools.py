import unittest,sys,importlib.util,struct
from pathlib import Path
from fractions import Fraction
from math import factorial
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bloom_inspector',P/'inspect_lowering.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=P/'native/sm80-fp16-b32-w4-fusion0'
PTX=(D/'kernel.ptx').read_text();TTIR=(D/'kernel.ttir').read_text()
class Checks(unittest.TestCase):
 def reject(self,p=PTX,t=TTIR):
  with self.assertRaises(ValueError):m.inspect(p,t)
 def inject(self,line,extra):return PTX.replace(line,line+'\n'+extra,1)
 def test_actual_profile(self):self.assertEqual(len(m.inspect(PTX,TTIR)['ptx_exponent_chains']),1)
 def test_wrong_log2e(self):self.reject(PTX.replace('0f3FB8AA3B','0f3F800000'))
 def test_no_subtract(self):self.reject(PTX.replace('sub.rn.f32','add.rn.f32'))
 def test_wrong_exp(self):self.reject(PTX.replace('ex2.approx.f32','ex2.approx.ftz.f32'))
 def test_wrong_division(self):self.reject(PTX.replace('div.full.f32','div.rn.f32'))
 def test_exp_output_overwrite(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['ex2'];self.reject(self.inject(e['text'],f"mov.b32 {e['dst']}, 0;"))
 def test_predicated_exp_output_overwrite(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['ex2'];self.reject(self.inject(e['text'],f"@%p999 mov.b32 {e['dst']}, 0;"))
 def test_sub_output_overwrite(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['subtract'];self.reject(self.inject(e['text'],f"mov.b32 {e['dst']}, 0;"))
 def test_mul_output_overwrite(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['scale_multiply'];self.reject(self.inject(e['text'],f"mov.b32 {e['dst']}, 0;"))
 def test_vector_destination_overwrite(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['ex2'];self.reject(self.inject(e['text'],f"ld.global.v2.b32 {{{e['dst']}, %r999}}, [%rd999];"))
 def test_unknown_write_rejected(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['ex2'];self.reject(self.inject(e['text'],f"custom.op %r999, {e['dst']};"))
 def test_predicated_producer(self):self.reject(PTX.replace('mul.rn.f32','@%p999 mul.rn.f32'))
 def test_scalar_control_branch(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['scale_multiply'];self.reject(self.inject(e['text'],'@%p999 bra $L__BB999;\n$L__BB999:'))
 def test_external_branch_to_debug_named_label(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['scale_multiply']
  bad='bra $L__tmp999;\n'+self.inject(e['text'],'$L__tmp999:')
  self.reject(bad)
 def test_unknown_control_in_scalar_span(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['scale_multiply'];self.reject(self.inject(e['text'],'custom.control;'))
 def test_div_control_explicitly_unproved(self):
  e=m.inspect(PTX,TTIR)['ptx_exponent_chains'][0]['ex2'];x=m.inspect(self.inject(e['text'],'@%p999 bra $L__BB999;\n$L__BB999:'),TTIR)
  self.assertEqual(x['ptx_division_chains'][0]['numerator_span']['status'],'CONTROL_OR_SYNCHRONIZATION_NOT_PROVED')
 def test_div_numerator_wrong(self):self.reject(PTX.replace('%r38, %r27, %r37','%r38, %r26, %r37'))
 def test_wrong_source_exp_input(self):self.reject(t=TTIR.replace('math.exp %row_minus_max_15','math.exp %row_13'))
 def test_wrong_source_reduce(self):self.reject(t=TTIR.replace('arith.maxnumf','arith.maximumf'))
 def test_independent_rational_cells(self):
  for x,lo,hi in [(Fraction(-6291459,8388608),31699971,31699973),(Fraction(-12582919,16777216),31699969,31699971)]:
   total=sum(x**i/factorial(i) for i in range(16));err=abs(x)**16*Fraction(17,factorial(16)*16)
   self.assertGreater(total-err,Fraction(lo,67108864));self.assertLess(total+err,Fraction(hi,67108864))
 def test_half_to_float_subtract_inputs(self):
  for word,target in [(0x8006,0xbf400006),(0x8007,0xbf400007)]:
   val=struct.unpack('<e',struct.pack('<H',word))[0];out=struct.unpack('<I',struct.pack('<f',val-0.75))[0];self.assertEqual(out,target)
 def test_scaled_collision_cpu(self):
  ff=lambda x:struct.unpack('<f',struct.pack('<I',x))[0];bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
  for x in [0xbf400006,0xbf400007]:self.assertEqual(bits(ff(x)*ff(0x3fb8aa3b)),0xbf8a7fb1)
if __name__=='__main__':unittest.main()
