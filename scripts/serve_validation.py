"""Run the actual Streamlit app against a loopback mock API and isolated data."""
from pathlib import Path
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import APP_ROOT
from tests.local_mock_api import LocalMockAPI

if __name__ == '__main__':
    directory=APP_ROOT/'reports'/'phase2'/'browser-data'
    with LocalMockAPI(port=8766,log_path=APP_ROOT/'reports'/'phase2'/'browser-api.json') as mock:
        env={**os.environ,'DEEPSEEK_API_KEY':'local-validation-only',
             'DEEPSEEK_BASE_URL':mock.base_url,'DEEPSEEK_DATA_DIR':str(directory),
             'DEEPSEEK_MAX_RETRIES':'0','RUN_PAID_API_TESTS':'0',
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(APP_ROOT/'app.py'),
                                  '--server.headless','true','--server.address','127.0.0.1',
                                  '--server.port','8503','--browser.gatherUsageStats','false'],env=env)
        try:
            print('Actual app: http://127.0.0.1:8503 ; API is LOOPBACK MOCK ONLY',flush=True)
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=10)
