from pathlib import Path
import json, os, secrets, subprocess, sys, time, urllib.request
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
URL='http://127.0.0.1:21461'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def health():
    try: return json.load(opener.open(URL+'/health',timeout=2))
    except Exception: return None
if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'status'
    if action=='start':
        if health(): print('already running');sys.exit(0)
        RUNTIME.mkdir(exist_ok=True)
        (RUNTIME/'stop-token').write_text(secrets.token_urlsafe(32))
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',USE_TF='0')
        with (RUNTIME/'service.log').open('ab') as log:
            p=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'service.py')],cwd=ROOT,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        (RUNTIME/'pid').write_text(str(p.pid))
        print('started',p.pid)
    elif action=='stop':
        if not health(): print('not listening');sys.exit(0)
        req=urllib.request.Request(URL+'/admin/shutdown',data=b'{}',headers={'Authorization':'Bearer '+(RUNTIME/'stop-token').read_text()})
        opener.open(req,timeout=5).close()
        for _ in range(50):
            time.sleep(.2)
            if not health(): print('graceful shutdown complete');break
        else: raise SystemExit('still running; force-stop was not attempted')
    else: print(json.dumps(health(),ensure_ascii=False))
