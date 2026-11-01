"""Explicit, resumable model setup. Downloads only; never imports an inference runtime."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

MODELS = {
    'qwen-asr': ('Qwen/Qwen3-ASR-1.7B', 'qwen', 'Qwen3-ASR-1.7B'),
    'qwen-4b': ('Qwen/Qwen3.5-4B', 'qwen', 'Qwen3.5-4B'),
    'whisper-v3': ('Systran/faster-whisper-large-v3', 'other', 'OpenAI/faster-whisper-large-v3'),
    'whisper-turbo': ('mobiuslabsgmbh/faster-whisper-large-v3-turbo', 'other', 'OpenAI/faster-whisper-large-v3-turbo'),
    'voxtral': ('mistralai/Voxtral-Mini-4B-Realtime-2602', 'other', 'Mistral/Voxtral-Mini-4B-Realtime-2602'),
    'translategemma': ('RedHatAI/translategemma-4b-it', 'other', 'Google/translategemma-4b-it'),
}


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def curl_base(args):
    result = [args.curl, '--fail', '--location', '--silent', '--show-error',
              '--connect-timeout', '30', '--retry', '5', '--retry-delay', '3']
    if args.proxy:
        result += ['--proxy', args.proxy]
    # Optional existing account credential. Never write it to the manifest/log.
    return result


def fetch(args, url, dest, resume=False):
    command = curl_base(args) + ['--output', str(dest)]
    if resume:
        command += ['--continue-at', '-']
    # Keep credentials off the command line (and curl's verbose logs).
    token = os.environ.get('HF_TOKEN', '')
    config = f'header = "Authorization: Bearer {token}"\n' if token else ''
    result = subprocess.run(command + ['--config', '-', url], input=config, text=True)
    if result.returncode:
        raise RuntimeError(f'curl failed ({result.returncode}) for {url.split("/resolve/")[0]}')


def selected_files(info):
    files = info['siblings']
    has_hf_weights = any(f['rfilename'].startswith('model') and f['rfilename'].endswith('.safetensors') for f in files)
    for f in files:
        name = f['rfilename']
        if '/' in name or name.startswith('.'):
            continue
        if name == 'consolidated.safetensors' and has_hf_weights:
            continue  # avoid storing the same weights in two layouts
        if name.endswith(('.json', '.txt', '.jinja', '.model', '.safetensors')) or name in ('model.bin', 'README.md', 'LICENSE', 'NOTICE'):
            yield f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-root', type=Path, default=Path('J:/Projects/LLM_Models/Qwen'))
    parser.add_argument('--other-root', type=Path, default=Path('K:/Projects/LLM_Models'))
    parser.add_argument('--manifest', type=Path, default=Path('config/local/speech-downloads.json'))
    parser.add_argument('--proxy', default='')
    parser.add_argument('--curl', default=shutil.which('curl.exe') or 'curl')
    parser.add_argument('--only', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text('utf-8')) if args.manifest.exists() else {'models': {}}
    for name in args.only:
        if name in manifest['models']:
            continue  # resume the pinned revision, not a moving main
        repo, root, relative = MODELS[name]
        meta = args.manifest.parent / f'{name}.metadata.json'
        fetch(args, f'https://huggingface.co/api/models/{repo}?blobs=true', meta)
        info = json.loads(meta.read_text('utf-8'))
        revision = info['sha']
        dest = (args.qwen_root if root == 'qwen' else args.other_root) / relative
        entries = []
        for f in selected_files(info):
            entries.append({'file': f['rfilename'], 'bytes': f['size'],
                            'sha256': f.get('lfs', {}).get('sha256'), 'git_blob': f.get('blobId'),
                            'path': str(dest / f['rfilename']),
                            'url': f'https://huggingface.co/{repo}/resolve/{revision}/{f["rfilename"]}'})
        manifest['models'][name] = {'repo': repo, 'revision': revision, 'files': entries}
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    entries = [f for n in args.only for f in manifest['models'][n]['files']]
    print(json.dumps({'files': len(entries), 'bytes': sum(f['bytes'] for f in entries),
                      'models': {n: manifest['models'][n]['revision'] for n in args.only}}), flush=True)
    if args.plan_only:
        return

    def download(entry):
        dest = Path(entry['path'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected, git = (entry['sha256'], False) if entry['sha256'] else (entry['git_blob'], True)
        def valid(p):
            return p.is_file() and p.stat().st_size == entry['bytes'] and expected and digest(p, git) == expected
        if valid(dest):
            return {'path': str(dest), 'verified': True, 'reused': True}
        if dest.exists():
            raise RuntimeError(f'Existing file has unexpected content; preserved: {dest}')
        part = dest.with_name(dest.name + '.partial')
        if not valid(part):
            fetch(args, entry['url'], part, resume=True)
        if not valid(part):
            raise RuntimeError(f'Integrity check failed: {part}')
        part.replace(dest)
        print('verified ' + str(dest), flush=True)
        return {'path': str(dest), 'verified': True, 'reused': False}

    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, e): e for e in entries}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'path': futures[future]['path'], 'error': str(exc)})
                print(str(exc), flush=True)
    report = {'completed_at': time.time(), 'verified': results, 'errors': errors}
    args.manifest.with_name('speech-download-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
