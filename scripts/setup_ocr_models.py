"""Download pinned official OCR GGUFs with SHA256 verification (explicit setup only)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import time
import argparse


def download(url, dest, expected=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    def digest(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(b)
        return h.hexdigest()
    if dest.exists() and expected and digest(dest) == expected:
        print('verified', dest, flush=True)
        return
    part = dest.with_suffix(dest.suffix + '.partial')
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req, timeout=90) as response:
                mode = 'ab' if offset and response.status == 206 else 'wb'
                with part.open(mode) as f:
                    while chunk := response.read(4 * 1024 * 1024):
                        f.write(chunk)
            actual = digest(part)
            if expected and actual != expected:
                raise ValueError(f'SHA256 mismatch: {dest.name}: {actual}')
            part.replace(dest)
            print('downloaded', dest, actual, flush=True)
            return
        except Exception as e:
            print('retry', dest.name, str(e), flush=True)
            if attempt == 4:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='K:/Projects/LLM_Models')
    args = parser.parse_args()
    root = Path(args.root)
    entries = []
    for repo, revision, names in [
        ('openbmb/MiniCPM-OCR-1.0-gguf', '674e62bf4ef7560ba502fe9b366a2780028aef20', ['MiniCPM-OCR-1_0-F16.gguf', 'mmproj-model-f16-4x.gguf']),
        ('openbmb/MiniCPM-OCR-Jev-gguf', '2554d7c9173b4cb767c4ecfc3e886e824978965c', ['MiniCPM-OCR-Jev-F16.gguf', 'mmproj-model-f16.gguf']),
    ]:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as r:
            info = json.load(r)
        for name in names + ['README.md']:
            f = next(x for x in info['siblings'] if x['rfilename'] == name)
            entries.append(dict(repo=repo, revision=revision, file=name, bytes=f.get('size'),
                                sha256=f.get('lfs', {}).get('sha256'),
                                url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}',
                                path=str(root / repo.split('/')[-1] / name)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, e['url'], e['path'], e['sha256']) for e in entries]
        for future in futures:
            future.result()
    (root / 'minicpm-ocr-manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
