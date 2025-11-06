#!/usr/bin/env python3
"""
Three multiprocessing patterns in one file:
  --mode pipe    : 2-process duplex dialog using Pipe
  --mode queue   : multi-producer/multi-consumer FIFO using Queue
  --mode shared  : zero-copy array via SharedMemory (NumPy view)

All demos are deterministic and print their results, so you can compare
performance/behavior quickly. Designed to be Windows-safe (spawn).
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from multiprocessing import Process, Pipe, Queue, set_start_method
from typing import List, Tuple

# ---------- Pipe demo: request/response over a duplex channel ----------

def _pipe_child(conn):
    """
    Child side of a duplex Pipe.
    Protocol:
      1) Receive a list of ints
      2) Send back a dict: {'sum': ..., 'squares': [...], 'pid': ...}
      3) Receive an acknowledgment string; print it; exit
    """
    try:
        data = conn.recv()  # blocks until parent sends
        # lightweight CPU work
        total = sum(data)
        squares = [x * x for x in data]
        conn.send({"sum": total, "squares": squares, "pid": os.getpid()})
        ack = conn.recv()
        # Show that we got the parent's final message
        print(f"[child:{os.getpid()}] ACK from parent: {ack}")
    finally:
        conn.close()


def run_pipe_demo():
    parent, child = Pipe(duplex=True)
    p = Process(target=_pipe_child, args=(child,))
    p.start()

    payload = list(range(1, 8))  # 1..7
    print(f"[parent:{os.getpid()}] sending: {payload}")
    parent.send(payload)

    resp = parent.recv()
    print(f"[parent] received: {resp}")
    parent.send("thanks, kiddo")   # one more hop the other way

    p.join(timeout=5)
    if p.is_alive():
        print("[parent] child did not exit in time")
    parent.close()


# ---------- Queue demo: fan-out/fan-in with many workers ---------------

def _queue_worker(q_in: Queue, q_out: Queue, wid: int):
    """
    Worker that:
      - reads ints from q_in
      - exits on sentinel None
      - writes (wid, x, f(x)) to q_out
    """
    while True:
        x = q_in.get()
        if x is None:
            break
        # pretend it's heavier work
        fx = (x * x) + 3 * x + 1  # simple quadratic
        q_out.put((wid, x, fx))


def run_queue_demo():
    from multiprocessing import cpu_count
    n_workers = max(2, min(4, cpu_count()))
    q_in, q_out = Queue(maxsize=64), Queue(maxsize=64)

    workers = [Process(target=_queue_worker, args=(q_in, q_out, i)) for i in range(n_workers)]
    for p in workers:
        p.start()

    tasks = list(range(10))  # 0..9
    for x in tasks:
        q_in.put(x)

    # stop signals
    for _ in workers:
        q_in.put(None)

    results = []
    for _ in tasks:
        results.append(q_out.get())

    for p in workers:
        p.join(timeout=5)

    # Sort by input x for readability
    results.sort(key=lambda t: t[1])
    print(f"[parent:{os.getpid()}] workers={n_workers}")
    for wid, x, fx in results:
        print(f"  worker {wid} computed f({x}) = {fx}")


# ---------- Shared memory demo: zero-copy NumPy array -------------------

def run_shared_demo():
    """
    Parent creates a NumPy array in shared memory. Child attaches by name,
    modifies a slice in place, parent observes the change without copies.
    """
    try:
        import numpy as np
        from multiprocessing import shared_memory
    except Exception as e:
        print("This demo requires numpy. Install it with: pip install numpy")
        raise

    # Parent creates data
    a = np.arange(16, dtype=np.int64)  # 0..15
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    try:
        # Create a view into the shared buffer and copy initial data
        a_sh = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        a_sh[:] = a  # initial state
        print(f"[parent:{os.getpid()}] initial: {a_sh.tolist()}")

        def _child(name: str, shape: Tuple[int, ...], dtype_str: str):
            import numpy as _np
            from multiprocessing import shared_memory as _sm
            shm_c = _sm.SharedMemory(name=name)
            try:
                arr = _np.ndarray(shape, dtype=_np.dtype(dtype_str), buffer=shm_c.buf)
                # In-place modify: double indices 4..11
                arr[4:12] *= 2
                # Also write a signature value at the end
                arr[-1] = 9999
                print(f"[child:{os.getpid()}] modified slice 4:12 and tail -> done")
            finally:
                shm_c.close()

        p = Process(target=_child, args=(shm.name, a_sh.shape, str(a_sh.dtype)))
        p.start()
        p.join(timeout=5)

        # Parent reads the same shared buffer, no copies
        print(f"[parent] after child: {a_sh.tolist()}")

    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()


# ---------- Main --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipe vs Queue vs SharedMemory demos")
    parser.add_argument("--mode", choices=["pipe", "queue", "shared"], required=True,
                        help="Which demo to run")
    args = parser.parse_args()

    # Windows-safe start method
    try:
        set_start_method("spawn")
    except RuntimeError:
        # Already set by another import; that’s fine.
        pass

    t0 = time.perf_counter()
    if args.mode == "pipe":
        run_pipe_demo()
    elif args.mode == "queue":
        run_queue_demo()
    else:
        run_shared_demo()
    dt = time.perf_counter() - t0
    print(f"[done] mode={args.mode} elapsed={dt:.3f}s")


if __name__ == "__main__":
    sys.exit(main())
