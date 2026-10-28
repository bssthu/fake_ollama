"""Synthetic, labeled OCR smoke fixtures. No personal screenshots are collected."""
from pathlib import Path
import base64
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def main():
    root = Path('.tmp/ocr-research/samples')
    root.mkdir(parents=True, exist_ok=True)
    cases = {
        'mixed': ('订单编号：OCR-2026-1029\n苹果 3 个，单价 12.50 元，总计 37.50 元。\nPlease verify invoice A-019: Total USD 123.45.', 30),
        'small': ('识别测试：未付款 ≠ 已付款；金额 108.08 元。\nCode: O0-I1-lL / Invoice 2026-10-29 / Qty 8\n请核对小字、数字、标点和中英文混排。', 18),
        'blurred': ('发票金额：￥1,280.50\n账户编号：AB-00917，数量：12，税率：13%\nDo not confuse 0 with O, or 1 with I.', 24),
    }
    for name, (text, size) in cases.items():
        image = Image.new('RGB', (1000, 250), 'white')
        draw = ImageDraw.Draw(image)
        draw.multiline_text((28, 20), text, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size), fill='black', spacing=18)
        if name == 'blurred':
            image = image.filter(ImageFilter.GaussianBlur(.8))
        image.save(root / (name+'.png'))
        (root / (name+'.txt')).write_text(text, encoding='utf-8')
        request = dict(image=base64.b64encode((root/(name+'.png')).read_bytes()).decode(), mode='compare',
                       engines=['minicpm-ocr','ppocr-v6','hunyuan-ocr-current'], verifier='minicpm-ocr-jev', max_tokens=1024)
        (root/(name+'.request.json')).write_text(json.dumps(request), encoding='utf-8')
    print('Created', len(cases), 'UTF-8 labeled fixtures')

if __name__ == '__main__':
    main()
