"""Run one decision with local weights or the released Hugging Face model."""
import argparse
import json
from pathlib import Path
from openjev.runtime.engine import DecisionEngine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='shenjunhao/OpenJev-4B')
    p.add_argument('--input',default=str(Path(__file__).with_name('decision.json')))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execution-mode',choices=['expanded','shared_prefix','auto'],default='expanded')
    a=p.parse_args()
    engine=DecisionEngine(a.model,device=a.device,execution_mode=a.execution_mode)
    print(json.dumps(engine.predict(json.loads(Path(a.input).read_text())),indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
