"""Create explicit local speech configuration; never enables inference by default."""
import json
from pathlib import Path


def configuration(root):
    python = str(root / 'services/speech/.venv/Scripts/python.exe')
    qwen_python = str(root / 'services/speech/qwen-asr/.venv/Scripts/python.exe')
    def spec(label, kind, path, required, note, **extra):
        return dict(label=label, kind=kind, path=path, required_files=required, note=note, **extra)
    return {
        'inference_enabled': False, 'python': python, 'ffmpeg': 'C:/shortcut/ffmpeg.exe',
        'default_asr': 'whisper-large-v3-turbo', 'default_translator': 'qwen3.5-2b',
        'client_timeout_seconds': 45, 'max_upload_bytes': 2 * 1024**3,
        'asr': {
            'qwen3-asr-1.7b': spec('Qwen3-ASR 1.7B', 'qwen_asr', 'J:/Projects/LLM_Models/Qwen/Qwen3-ASR-1.7B',
                ['config.json', 'model.safetensors.index.json', 'tokenizer_config.json'],
                '推荐 16 秒窗口改善自动语种判断；短句、动画声线和人名仍可能出错。官方流式接口需要 vLLM。', python=qwen_python),
            'whisper-large-v3': spec('Whisper large-v3', 'whisper', 'K:/Projects/LLM_Models/OpenAI/faster-whisper-large-v3',
                ['config.json', 'model.bin', 'tokenizer.json'], '逐段自动检测；同段内快速切换语种时可能漏句。'),
            'whisper-large-v3-turbo': spec('Whisper large-v3 turbo', 'whisper', 'K:/Projects/LLM_Models/OpenAI/faster-whisper-large-v3-turbo',
                ['config.json', 'model.bin', 'tokenizer.json'], '默认选择；本机中英切换和日语样本已实测，速度较快。'),
            'voxtral-mini-4b-realtime-2602': spec('Voxtral Mini 4B Realtime 2602', 'voxtral', 'K:/Projects/LLM_Models/Mistral/Voxtral-Mini-4B-Realtime-2602',
                ['config.json', 'model.safetensors', 'tekken.json'], '原生流式输入输出；可调整识别延迟和上下文上限，不提供可靠语种标签。'),
        },
        'translators': {
            'qwen3.5-2b': spec('Qwen3.5 2B · 复用本机权重', 'qwen_text', 'J:/Projects/LLM_Models/Qwen/Qwen3.5-2B',
                ['config.json', 'model.safetensors.index.json', 'tokenizer.json'], '自动理解混合语言；保留最近四段翻译上下文。'),
            'qwen3.5-4b': spec('Qwen3.5 4B', 'qwen_text', 'J:/Projects/LLM_Models/Qwen/Qwen3.5-4B',
                ['config.json', 'model.safetensors.index.json', 'tokenizer.json'], '自动理解混合语言；关闭思考输出以减少等待。'),
            'translategemma-4b': spec('TranslateGemma 4B', 'translategemma', 'K:/Projects/LLM_Models/Google/translategemma-4b-it',
                ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'chat_template.jinja'],
                '内部按文本片段推断源语言；短句及句内混语仍有检测误差，失败时保留原文。'),
        },
    }


def main():
    root = Path(__file__).resolve().parent.parent
    path = root / 'config/local/speech.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit('Existing speech config preserved; edit it explicitly if needed.')
    path.write_text(json.dumps(configuration(root), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    gateway_path = root / 'config.json'
    gateway = json.loads(gateway_path.read_text('utf-8'))
    gateway['playground_speech_config'] = str(path)
    gateway_path.write_text(json.dumps(gateway, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Configured {path}; inference_enabled=false. No service restart or model load.')


if __name__ == '__main__':
    main()
