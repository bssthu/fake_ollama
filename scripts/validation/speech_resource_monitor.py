"""Sample only a specified validation server's process tree; never imports torch."""
import argparse
import json
from pathlib import Path
import time

import psutil


def main(args):
    root = psutil.Process(args.pid)
    report = {'pid':args.pid,'created':root.create_time(),'samples':[]}
    started = time.monotonic()
    while root.is_running() and time.monotonic()-started < args.seconds:
        rss = private = 0
        processes = []
        for process in [root,*root.children(recursive=True)]:
            try:
                memory = process.memory_info()
                rss += memory.rss
                private += getattr(memory,'private',memory.vms)
                processes.append(process.pid)
            except psutil.NoSuchProcess:
                pass
        report['samples'].append({'time':time.time(),'elapsed':round(time.monotonic()-started,1),
                                  'rss_mib':round(rss/1048576,1),'private_mib':round(private/1048576,1),
                                  'pids':processes})
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        if args.until.exists():
            state = json.loads(args.until.read_text('utf-8'))
            if state.get('finished') or state.get('error'): break
        time.sleep(10)
    print('RESOURCE_MONITOR_DONE '+str(len(report['samples']))+' samples',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--seconds',type=int,default=1500)
    parser.add_argument('--output',type=Path,default=Path('.tmp/speech-live/resources.json'))
    parser.add_argument('--until',type=Path,default=Path('.tmp/speech-live/browser/results.json'))
    main(parser.parse_args())
