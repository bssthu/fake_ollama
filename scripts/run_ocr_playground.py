"""Serve only the local Playground, without opening the configured API/LAN listeners."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn
from fake_ollama.config import load_settings
from fake_ollama.server import create_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--port', type=int, default=21431)
    args = parser.parse_args()
    settings = load_settings(args.config)
    settings = settings.model_copy(update={'playground_enabled': True, 'playground_host': '127.0.0.1',
                                           'playground_port': args.port, 'admin_enabled': False,
                                           'dashboard_enabled': False})
    uvicorn.run(create_app(settings), host='127.0.0.1', port=args.port, proxy_headers=False)
