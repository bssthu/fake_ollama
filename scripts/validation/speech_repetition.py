"""Replay the exact retained translation failures and preceding context on GPU."""
import asyncio
import json
from pathlib import Path
import tempfile

from fake_ollama.speech_runtime import ModelWorker

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'.tmp/speech-live'


async def main():
    config=json.loads((ROOT/'config/local/speech.json').read_text('utf-8'))
    rows=json.loads((WORK/'soak-recovered-rows.json').read_text('utf-8'))
    history=[]
    cases=[]
    for row in rows:
        if row['state']=='translation_error': cases.append((row,list(history)))
        else:
            history.append({'source':row['text'][-1200:],'translation':row['translation'][-1200:]})
            history=history[-4:]
    report=[]
    with tempfile.TemporaryDirectory(prefix='speech-repetition-') as temp:
        worker=ModelWorker(Path(temp)/'translator',config['translators']['qwen3.5-2b'],config['python'])
        try:
            await worker.call({'op':'load'})
            for row,context in cases:
                result=await worker.call({'op':'translate','text':row['text'],'target':'zh','context':context})
                assert result['translation'] and result['seconds']<10,result
                report.append({'id':row['id'],'text':row['text'],**result})
                print('REPETITION_CASE '+json.dumps(report[-1],ensure_ascii=False),flush=True)
        finally:
            await worker.stop()
    (WORK/'repetition-regression.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPETITION_REGRESSION_PASS '+str(len(report)),flush=True)


if __name__=='__main__': asyncio.run(main())
