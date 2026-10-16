"""Private group anchor. Only this living group leader signals its own group.

Invoked by process_utils, never by an end user. A control pipe avoids sending
signals to a recycled PID/PGID after a launcher exits. EOF also cleans the group.
"""
import json
import os
import select
import signal
import subprocess
import sys
import time


def main():
    options = json.loads(sys.argv[1])
    # A caught (rather than ignored) handler resets to default in exec children.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    child = subprocess.Popen(options["command"], shell=options["shell"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    reported = False
    while True:
        code = child.poll()
        if code is not None and not reported:
            print(json.dumps({"returncode": code}), flush=True)
            reported = True
        readable, _, _ = select.select([sys.stdin], [], [], .02)
        if readable:
            line = sys.stdin.readline()
            grace = float(json.loads(line)["stop"]) if line else .5
            os.killpg(os.getpgrp(), signal.SIGTERM)
            time.sleep(max(0, min(grace, 10)))
            # Self is still the group anchor here; this group ID cannot refer
            # to a different generation. Include stubborn grandchildren.
            os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == "__main__":
    main()
