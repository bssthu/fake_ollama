"""Sequential real-model smoke benchmark; saves raw predictions, not leaderboard claims."""
import asyncio
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.ocr_pipeline import OCRRequest
from fake_ollama.ocr_worker import run


async def main():
    config=json.loads(Path('config/local/ocr.json').read_text(encoding='utf-8'))
    root=Path('.tmp/ocr-research/samples')
    for name in ['mixed','small','blurred']:
        payload=json.loads((root/(name+'.request.json')).read_text(encoding='utf-8'))
        payload['engines']=list(config['engines'])
        result=await run(config,OCRRequest(**payload))
        (root/(name+'.final.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name, [(r['engine'],r['score'],r.get('error')) for r in result['results']],flush=True)
    verifier = config['verifiers']['minicpm-ocr-jev']
    for case_path in Path(verifier['renderer_source']).glob('examples/*/case.json'):
        case=json.loads(case_path.read_text(encoding='utf-8'))
        payload={'mode':'score','image':base64.b64encode((case_path.parent/'block.png').read_bytes()).decode(),
                 'category':case['category'],'candidates':[c['text'] for c in case['candidates']], 'verifier':'minicpm-ocr-jev'}
        result=await run(config,OCRRequest(**payload))
        result['official_reference']=case['observed_prob_yes']
        (root/(case['case_id']+'.score.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(case['case_id'],[(r['score'],r.get('verification_error')) for r in result['results']],flush=True)

if __name__=='__main__':asyncio.run(main())
