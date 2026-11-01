"""Parallel HTTP ranges into one resumable file; bounded temporary ranges, no model copies."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import threading
import time

from setup_speech_models import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('config/local/speech-downloads.json'))
    p.add_argument('--proxy', default='http://127.0.0.1:6268')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--curl', default='C:/Program Files/Git/mingw64/bin/curl.exe')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text('utf-8'))
    entries = [e for model in manifest['models'].values() for e in model['files']]
    locks = {e['path']: threading.Lock() for e in entries}
    plans = {}
    jobs = []
    for entry in entries:
        dest = Path(entry['path'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue  # verified below
        part = dest.with_name(dest.name + '.partial')
        journal = dest.with_name(dest.name + '.ranges.json')
        if journal.exists():
            plan = json.loads(journal.read_text('utf-8'))
            if plan['url'] != entry['url'] or plan['bytes'] != entry['bytes']:
                raise ValueError(f'Pinned source changed for {dest}')
        else:
            prefix = part.stat().st_size if part.exists() else 0
            if prefix > entry['bytes']:
                raise ValueError(f'Partial file too large: {part}')
            plan = {'url': entry['url'], 'bytes': entry['bytes'], 'prefix': prefix, 'complete': []}
            journal.write_text(json.dumps(plan), encoding='utf-8')
        plans[entry['path']] = plan
        for start in range(plan['prefix'], entry['bytes'], 32 * 1024**2):
            if start not in plan['complete']:
                jobs.append((entry, start, min(entry['bytes'] - 1, start + 32 * 1024**2 - 1)))
    print(f'{len(jobs)} remaining ranges; {args.workers} transfers; pinned sources; full hashes at completion', flush=True)

    def transfer(job):
        entry, start, end = job
        dest = Path(entry['path'])
        part = dest.with_name(dest.name + '.partial')
        chunk = dest.with_name(f'{dest.name}.range-{start}.partial')
        header = chunk.with_suffix('.headers')
        for attempt in range(4):
            url = entry.get('download_url', entry['url'])
            transport = ['--noproxy', '*'] if 'modelscope.cn/' in url else ['--proxy', args.proxy]
            command = [args.curl, *transport, '-sSfL', '--connect-timeout', '20', '--max-time', '180',
                       '--retry', '2', '--retry-delay', '2', '--range', f'{start}-{end}',
                       '-D', str(header), '-o', str(chunk), url + f'?download=true&part={start}']
            result = subprocess.run(command, capture_output=True, text=True)
            content_range = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', header.read_text('utf-8', errors='replace') if header.exists() else '', re.I)
            correct_range = content_range and tuple(map(int, content_range[-1])) == (start, end, entry['bytes'])
            whole_small_file = start == 0 and end + 1 == entry['bytes'] and not content_range
            if result.returncode == 0 and chunk.stat().st_size == end - start + 1 and (correct_range or whole_small_file):
                break
            if attempt == 3:
                raise RuntimeError(f'Range download failed: {dest.name}:{start} curl={result.returncode}')
            time.sleep(2)
        with locks[entry['path']]:
            with part.open('r+b' if part.exists() else 'w+b') as stream, chunk.open('rb') as source:
                stream.seek(start)
                for block in iter(lambda: source.read(4 * 1024**2), b''):
                    stream.write(block)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            plan = plans[entry['path']]
            plan['complete'].append(start)
            journal = dest.with_name(dest.name + '.ranges.json')
            temporary = journal.with_suffix('.tmp')
            temporary.write_text(json.dumps(plan), encoding='utf-8')
            temporary.replace(journal)
        chunk.unlink()
        header.unlink(missing_ok=True)
        return end - start + 1

    errors, received = [], 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transfer, job): job for job in jobs}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                received += future.result()
            except Exception as exc:
                errors.append(str(exc)); print(str(exc), flush=True)
            if i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} ranges; received {received/1e9:.2f} GB; {received/max(1,time.monotonic()-started)/1e6:.1f} MB/s', flush=True)
    verified = []
    for entry in entries:
        dest = Path(entry['path'])
        path = dest if dest.exists() else dest.with_name(dest.name + '.partial')
        expected, git = (entry['sha256'], False) if entry['sha256'] else (entry['git_blob'], True)
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path, git) != expected:
            errors.append(f'Full hash mismatch or incomplete: {dest}'); continue
        if path != dest:
            path.replace(dest)
        dest.with_name(dest.name + '.ranges.json').unlink(missing_ok=True)
        verified.append({'path': str(dest), 'bytes': entry['bytes'], 'verified': True, 'sha256': entry['sha256']})
        print('verified ' + str(dest), flush=True)
    report = {'completed_at': time.time(), 'verified': verified, 'errors': errors}
    args.manifest.with_name('speech-download-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
