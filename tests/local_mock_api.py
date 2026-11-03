"""Loopback-only OpenAI HTTP stub. Never contacts a provider or reads a real key."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import hashlib
import io
import json
import re
import threading
import time
from PIL import Image


class LocalMockAPI:
    def __init__(self, port=0, log_path=None):
        self.requests = []
        self.summaries = []
        self.log_path = log_path
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'local mock API')

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                request = json.loads(raw)
                owner.requests.append(request)
                messages = request.get('messages', [])
                system = str(messages[0].get('content', '')) if messages else ''
                texts, images, bad_roles = [], [], []
                for msg in messages:
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        texts.append(content)
                    else:
                        for block in content:
                            if block.get('type') == 'text':
                                texts.append(block['text'])
                            elif block.get('type') == 'image_url':
                                if msg['role'] != 'user':
                                    bad_roles.append(msg['role'])
                                data = base64.b64decode(block['image_url']['url'].split(',', 1)[1])
                                with Image.open(io.BytesIO(data)) as image:
                                    rgb = image.convert('RGB')
                                    pixels = list(rgb.resize((200,200)).getdata())
                                    blue = sum(b > 100 and b > r*1.5 and b > g*1.2 for r,g,b in pixels)
                                    images.append({'sha256':hashlib.sha256(data).hexdigest(),
                                                   'size':list(rgb.size), 'blue_pixels_sample':blue,
                                                   'bytes':len(data)})
                text = '\n'.join(texts)
                answer = '[本地模拟 API] 已收到问题，链路验证完成。'
                if images:
                    answer = '[本地模拟 API] 蓝色柱最高；这只是合成样例的模拟响应。'
                if '[S1]' in text:
                    answer += ' [S1]'
                if '路由模块' in system:
                    answer = json.dumps({'route':'kb','reason':'local validation'})
                elif '相关性评判器' in system:
                    indices = sorted(set(int(x) for x in re.findall(r'\[(\d+)\] 来源:',text)))
                    answer = json.dumps({'results':[{'idx':i,'relevant':True,'score':1} for i in indices]})
                elif '答案审核器' in system:
                    answer = json.dumps({'pass':True,'final_answer':answer,'note':'mock verification'},ensure_ascii=False)
                elif 'Judge if evidence' in system:
                    answer = json.dumps({'sufficient':True,'missing_query':'','reason':'mock coverage'})
                elif 'need_web' in system:
                    answer = '{"need_web":false}'
                elif 'Break a research question' in system:
                    answer = '{"queries":["Aster and Boreal controlled results"]}'
                elif '改写' in system and '"q"' in system:
                    answer = '{"q":"synthetic local validation"}'
                elif 'rewritten' in system:
                    answer = '{"rewritten":"Aster evidence"}'
                elif '"intent"' in system:
                    answer = '{"intent":"qa","note":"local validation"}'
                elif '"explanation"' in system:
                    answer = json.dumps({'explanation':'[本地模拟 API] 已按真实本地数据展示统计表。',
                                         'chart':None,'table':{'op':'head','n':5}},ensure_ascii=False)
                summary = {'sequence':len(owner.requests),'model':request.get('model'),
                           'request_bytes':len(raw),'images':images,'invalid_image_roles':bad_roles}
                owner.summaries.append(summary)
                if owner.log_path:
                    owner.log_path.parent.mkdir(parents=True,exist_ok=True)
                    owner.log_path.write_text(json.dumps(owner.summaries,ensure_ascii=False,indent=2))
                response = {'id':'local-mock','object':'chat.completion','created':int(time.time()),
                            'model':request.get('model'), 'choices':[{'index':0,'finish_reason':'stop',
                            'message':{'role':'assistant','content':answer}}],
                            'usage':{'prompt_tokens':12,'completion_tokens':8,'total_tokens':20}}
                body = json.dumps(response,ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.server = ThreadingHTTPServer(('127.0.0.1',port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
