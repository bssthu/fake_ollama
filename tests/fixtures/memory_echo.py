"""Harmless child for exercising the real owned IPC transport."""
from fake_ollama.memory_worker import Channel
from fake_ollama.privacy import privacy_mode_enabled

channel = Channel.connect()
try:
    while True:
        value, binary = channel.receive()
        if value.get('write'):
            try:
                with open(value['write'], 'wb') as stream:
                    stream.write(b'public-fixture')
                denied = False
            except PermissionError:
                denied = True
            channel.send({'result': {'denied': denied}})
        elif value.get('reverse'):
            response, data = channel.event({'kind': 'read', 'length': 4})
            channel.send({'result': {'data': data.decode()}})
        else:
            channel.send({'result': {'size': len(binary), 'privacy': privacy_mode_enabled()}})
except (EOFError, OSError):
    pass
finally:
    channel.close()
