"""Install the pinned multilingual decision model alongside fake_ollama."""
from pathlib import Path
import hashlib, json, urllib.request, zipfile, subprocess, sys
ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
SOURCE_REVISION = '1e28ac20c0896b1c37a744cd11f740eb98f8b178'
MODEL_REVISION = '82d57fc4f2d1be3d2caac494045f2ec51d0842f3'
def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix(target.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=120) as stream, part.open('wb') as out:
            while data := stream.read(1024*1024): out.write(data)
        part.replace(target)
if __name__ == '__main__':
    RUNTIME.mkdir(exist_ok=True)
    archive = RUNTIME / 'source.zip'
    fetch(f'https://codeload.github.com/NandhaKishorM/laya/zip/{SOURCE_REVISION}', archive)
    with zipfile.ZipFile(archive) as z: z.extractall(RUNTIME)
    repo = RUNTIME / f'laya-{SOURCE_REVISION}'
    api = f'https://huggingface.co/api/models/convaiinnovations/laya-multilingual/revision/{MODEL_REVISION}?blobs=true'
    metadata = json.load(urllib.request.urlopen(api, timeout=30))
    hashes = {}
    for entry in metadata['siblings']:
        name = entry['rfilename']
        if name == '.gitattributes': continue
        target = RUNTIME / 'model' / name
        fetch(f'https://huggingface.co/convaiinnovations/laya-multilingual/resolve/{MODEL_REVISION}/{name}', target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if entry.get('lfs'): assert digest == entry['lfs']['sha256'], f'checksum mismatch: {name}'
        hashes[name] = digest
        print('verified', name, target.stat().st_size, flush=True)
    (ROOT/'model-lock.json').write_text(json.dumps({'source_revision':SOURCE_REVISION,'model_revision':MODEL_REVISION,'files':hashes},indent=2)+'\n')
    print('DOWNLOAD_COMPLETE', flush=True)
