"""Public synthetic OCR fixture over the real strict-privacy listener."""
import base64
import argparse
import io
import json
import httpx
from PIL import Image, ImageDraw, ImageFont

image = Image.new('RGB', (720, 150), 'white')
ImageDraw.Draw(image).text((20, 35), 'PRIVACY TEST 123', fill='black', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 45))
buffer = io.BytesIO()
image.save(buffer, format='PNG')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine', default='ppocr-v6')
args = parser.parse_args()
payload = {'image': base64.b64encode(buffer.getvalue()).decode(), 'mode': 'recognize',
           'engines': [args.engine], 'keep_loaded': True}
with httpx.Client(base_url='http://127.0.0.1:21439', timeout=180, trust_env=False) as client:
    try:
        catalog = client.get('/playground/api/ocr/models').json()
        assert catalog['privacy']['enabled']
        for i in range(2):
            response = client.post('/playground/api/ocr/run', json=payload)
            response.raise_for_status()
            result = response.json()
            assert 'PRIVACY' in json.dumps(result)
            assert not any(row.get('error') for row in result['results'])
            print(json.dumps({'private_ocr': True, 'iteration': i, 'models': result['loaded_models'],
                              'events': result['runtime_events']}), flush=True)
    finally:
        response = client.post('/playground/api/ocr/release')
        response.raise_for_status()
        assert not response.json()['worker_running']
        print(json.dumps({'ocr_released': True}), flush=True)
