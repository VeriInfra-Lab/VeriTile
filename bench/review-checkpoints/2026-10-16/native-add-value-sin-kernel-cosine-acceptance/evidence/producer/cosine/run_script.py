from pathlib import Path
import sys,subprocess,hashlib,json,datetime
W=Path(__file__).resolve().parent
script=sys.argv[1];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
D=W/'evidence/script-attempts'/stamp;D.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['propose.py','authority_adapter.py','check_adapter.py','freeze.py',script]:
 p=W/name
 if p.is_file():(D/(name+'.txt')).write_bytes(p.read_bytes())
with (D/'console.log').open('w') as f:r=subprocess.run([sys.executable,script,*sys.argv[2:]],cwd=W,stdout=f,stderr=subprocess.STDOUT)
(D/'record.json').write_text(json.dumps({'script':script,'exit_code':r.returncode,'artifacts':{p.name:sha(p) for p in D.iterdir() if p.is_file()}},indent=2)+'\n')
print((D/'console.log').read_text());print('record='+str(D/'record.json'));sys.exit(r.returncode)
