import time
import logging
import json
from datetime import datetime


logger = logging.getLogger('telemetry')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('telemetry.log')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)




def log_request(question: str, pathway: str, latency_s: float, extra: dict | None = None):
rec = {
'timestamp': datetime.utcnow().isoformat() + 'Z',
'question_len': len(question),
'pathway': pathway,
'latency_s': latency_s,
}
if extra:
rec.update(extra)
logger.info(json.dumps(rec))