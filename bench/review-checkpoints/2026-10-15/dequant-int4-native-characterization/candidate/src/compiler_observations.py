#!/usr/bin/env python3
"""Extract textual compiler observations; this is not a semantic equivalence prover."""
from pathlib import Path
import argparse,hashlib,json,re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def observations(out):
    report=json.loads((out/'report.json').read_text());rows=[]
    for r in report['records']:
        label=r['label'];ttir=out/(label+'.ttir');ptx=out/(label+'.ptx')
        irlines=ttir.read_text().splitlines();plines=ptx.read_text().splitlines()
        dots=[{'line':i+1,'text':s.strip()} for i,s in enumerate(irlines) if '= tt.dot ' in s]
        halffma=[{'line':i+1,'text':s.strip()} for i,s in enumerate(plines) if re.search(r'\bfma\.rn\.f16(?:x2)?\s',s)]
        halfmul=[{'line':i+1,'text':s.strip()} for i,s in enumerate(plines) if re.search(r'\bmul(?:\.rn)?\.f16(?:x2)?\s',s)]
        halfsub=[{'line':i+1,'text':s.strip()} for i,s in enumerate(plines) if re.search(r'\bsub(?:\.rn)?\.f16(?:x2)?\s',s)]
        tensor=[{'line':i+1,'text':s.strip()} for i,s in enumerate(plines) if 'mma.sync.aligned.' in s or 'wgmma.mma_async.' in s]
        rows.append({'label':label,'constexprs':r['constexprs'],'options':r['options'],'ttir_sha256':sha(ttir),'ptx_sha256':sha(ptx),
          'tt_dot_occurrences':dots,'ptx_half_fma_occurrences':halffma,'ptx_half_mul_occurrences':halfmul,'ptx_half_sub_occurrences':halfsub,'ptx_tensor_mma_occurrences':tensor})
    return {'schema':1,'report_sha256':sha(out/'report.json'),'native_execution':False,
      'interpretation':'Textual observations only. tt.dot takes a live SSA accumulator in all eight captured loops. Fusion-on has scalar half FMA; fusion-off retains half multiply/subtract. Both use tensor MMA lowering. These facts do not prove a serial FP32 FMA order for MMA/WGMMA or native equality to any software reference.',
      'records':rows}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(observations(a.capture),indent=2)+'\n')
