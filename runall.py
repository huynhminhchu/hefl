import subprocess
import threading
from typing import List

# Test case: k = 2, k = 4, k = 6
# Each case test with r = 3, 5, 7

k = 2
r = 3


def call_process(k, i, r):
    with open(f"log_k{k}_r{r}_client{i}.txt", "w", 1) as f:
        p = subprocess.Popen(
            ["python", "multi_client.py", str(k), str(i), str(r)],
            shell=False,
            universal_newlines=True,
            stdout=f,
        )
        p.wait()
        f.flush()
        f.close()


threads_list: List[threading.Thread] = []
for i in range(k):
    t = threading.Thread(target=call_process, args=(k, i, r))
    t.start()
    threads_list.append(t)

for thread in threads_list:
    thread.join()
