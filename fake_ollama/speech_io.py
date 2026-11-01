"""Private speech mailbox I/O with bounded Windows sharing-conflict retries."""
import json
import time


def retry_io(operation):
    # Polling readers and Windows scanners can briefly deny rename/open/delete.
    # Retrying preserves atomic publication; a persistent failure still surfaces.
    for attempt in range(51):
        try:
            return operation()
        except PermissionError:
            if attempt == 50:
                raise
            time.sleep(.01)


def read_json(path):
    return json.loads(retry_io(lambda: path.read_text('utf-8')))


def unlink(path):
    retry_io(lambda: path.unlink(missing_ok=True))


def atomic_json(path, value):
    part = path.with_suffix('.partial')
    retry_io(lambda: part.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8'))
    retry_io(lambda: part.replace(path))
