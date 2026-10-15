from pathlib import Path
import hashlib,json,struct,sys,types

def digest(data):return hashlib.sha256(data).hexdigest()

def sha(p):return digest(Path(p).read_bytes())

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def load(p):return json.loads(p.read_text())

def need(ok,msg):
    if not ok:raise ValueError(msg)

def checked_module(path,name,expected):
    data=path.read_bytes();need(digest(data)==expected,'checked source bytes changed: '+str(path))
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(data,str(path),'exec'),m.__dict__)
    return m

def serial(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if hasattr(v,'_asdict'):return serial(v._asdict())
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return repr(v)

def compiler_environment(archs):
    import triton,triton.backends.nvidia.compiler as nc
    root=Path(triton.__file__).resolve().parent
    files={p.resolve() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    files.add(Path(sys.executable).resolve());tools={}
    for arch in archs:
        tool=Path(nc.get_ptxas(arch).path).resolve();files.add(tool);tools[str(arch)]={'ptxas_path':str(tool),'version':nc.get_ptxas_version(arch)}
    libdevice=Path(nc.knobs.nvidia.libdevice_path or root/'backends/nvidia/lib/libdevice.10.bc').resolve();files.add(libdevice)
    return {'file_sha256':{str(p):sha(p) for p in sorted(files)},'ptxas':tools,'libdevice':str(libdevice),'triton':triton.__version__,
      'boundary':'Actual Triton package, resolved ptxas/libdevice, Python executable before/after. OS and all host shared libraries are not hermetically captured.'}

def save_artifacts(kernel,out,label):
    files={}
    for kind,value in kernel.asm.items():
        need(kind.replace('_','').isalnum(),'unsafe artifact name')
        if not isinstance(value,(str,bytes)):continue
        q=out/(label+'.'+kind);q.write_bytes(value.encode() if isinstance(value,str) else value);files[q.name]=sha(q)
    need(all(label+'.'+x in files for x in ['ttir','ttgir','llir','ptx','cubin']),'missing compiler artifacts')
    metadata=serial(kernel.metadata);need(isinstance(metadata,dict),'metadata field names missing')
    return {'artifact_sha256':files,'resolved_metadata':metadata}

def raw_bytes(words,width):return struct.pack('<'+('H' if width==16 else 'I')*len(words),*(v&((1<<width)-1) for v in words))

def save_words(out,label,name,words,width):
    q=out/(label+'-'+name+'.bin');q.write_bytes(raw_bytes(words,width));return {'path':q.name,'sha256':sha(q),'word_bits':width,'words':len(words),'byte_order':'little'}
