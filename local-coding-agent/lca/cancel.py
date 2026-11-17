"""Cooperative cancellation shared between the UI thread, the agent loop, providers and tools."""

from __future__ import annotations

import threading
from collections.abc import Callable


class Cancelled(Exception):
    """Raised when the user cancels the running task."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception:  # a failing cleanup must not stop the others
                pass

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled()

    def wait(self, timeout: float) -> bool:
        """Sleep up to `timeout` seconds; returns True if cancelled meanwhile."""
        return self._event.wait(timeout)

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a callback (e.g. closing an HTTP stream). Returns a function that unregisters it."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                registered = True
            else:
                registered = False
        if not registered:
            callback()

        def unregister() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unregister
