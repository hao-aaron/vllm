# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred collective RPCs: a worker method returns a Future, the RPC is acked at
once, and each worker reports the Future's result later on a side channel."""

import json
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

DEFERRED_KWARG = "_vllm_deferred"


def gather_futures(futures: Sequence[Future]) -> Future[list]:
    """One Future for all of `futures`: their results, or the first error."""
    out: Future[list] = Future()
    remaining = [len(futures)]
    lock = threading.Lock()

    def done(_: Future) -> None:
        with lock:
            remaining[0] -= 1
            last = remaining[0] == 0
        if last and not out.done():
            for f in futures:
                if (e := f.exception()) is not None:
                    out.set_exception(e)
                    return
            out.set_result([f.result() for f in futures])

    if not futures:
        out.set_result([])
    for f in futures:
        f.add_done_callback(done)
    return out


class DeferredResults:
    """Executor side: one PULL socket and thread resolving one Future per call."""

    def __init__(self) -> None:
        ip = get_ip()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PULL)
        if ":" in ip:
            self._sock.setsockopt(zmq.IPV6, 1)
            ip = f"[{ip}]"
        port = self._sock.bind_to_random_port(f"tcp://{ip}")
        self.address = f"tcp://{ip}:{port}"
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[Future, int, dict[int, Any]]] = {}
        self._next_token = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="DeferredRPCResults"
        )
        self._thread.start()

    def register(self, num_workers: int) -> tuple[int, Future[list]]:
        """Register a call before it is sent, so a fast reply can't be lost."""
        future: Future[list] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Executor is shut down")
            token = self._next_token
            self._next_token += 1
            self._pending[token] = (future, num_workers, {})
        return token, future

    def discard(self, token: int) -> None:
        with self._lock:
            self._pending.pop(token, None)

    def close(self, reason: str) -> None:
        """Fail every pending call; replies that arrive later are dropped."""
        with self._lock:
            self._closed = True
            pending, self._pending = self._pending, {}
        for future, _, _ in pending.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        try:
            while not self._closed:
                if not poller.poll(timeout=200):
                    continue
                token, rank, ok, payload = json.loads(self._sock.recv())
                with self._lock:
                    entry = self._pending.get(token)
                    if entry is None:
                        continue
                    future, num_workers, results = entry
                    results[rank] = (ok, payload)
                    if len(results) < num_workers and ok:
                        continue
                    del self._pending[token]
                if ok:
                    future.set_result([results[r][1] for r in sorted(results)])
                else:
                    future.set_exception(
                        RuntimeError(f"Worker {rank} failed with error '{payload}'")
                    )
        finally:
            self._sock.close(linger=0)
            self._ctx.term()


class DeferredSender:
    """Worker side: report a Future's outcome to the executor when it completes."""

    def __init__(self, rank: int) -> None:
        self._rank = rank
        self._ctx: zmq.Context | None = None
        self._socks: dict[str, zmq.Socket] = {}
        self._lock = threading.Lock()

    def send_when_done(self, future: Future, address: str, token: int) -> None:
        future.add_done_callback(lambda f: self._send(f, address, token))

    def _send(self, future: Future, address: str, token: int) -> None:
        try:
            payload, ok = future.result(), True
        except BaseException as e:
            logger.exception("Deferred worker call failed")
            payload, ok = str(e), False
        # JSON: results here are None; richer results would need vLLM's msgpack.
        msg = json.dumps((token, self._rank, ok, payload)).encode()
        # Done callbacks run on whichever thread completes the Future.
        with self._lock:
            if self._ctx is None:
                self._ctx = zmq.Context()
            sock = self._socks.get(address)
            if sock is None:
                sock = self._socks[address] = self._ctx.socket(zmq.PUSH)
                sock.connect(address)
            sock.send(msg)
