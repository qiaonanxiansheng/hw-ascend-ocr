"""
Process-level ACL (AscendCL) environment management.

AscendCL resources (`acl.init` / `acl.finalize`) should be initialized and
finalized exactly once per process. This module provides a reference-counted
singleton so multiple `AscendModel` instances can coexist safely.
"""

import atexit
import threading
from typing import Optional

from .exceptions import YuntuAscendOCRError


try:
    import acl
except ImportError as exc:  # pragma: no cover - only available on Ascend hardware
    acl = None
    _ACL_IMPORT_ERROR = exc


class _ACLEnvironment:
    """Thread-safe, reference-counted ACL environment."""

    _instance: Optional["_ACLEnvironment"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "_ACLEnvironment":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._ref_count = 0
                    cls._instance._device_contexts = {}
        return cls._instance

    def _ensure_acl(self) -> None:
        if acl is None:
            raise YuntuAscendOCRError(
                "acl package is not available. "
                "Please run on an Ascend environment with the CANN toolkit installed."
            ) from _ACL_IMPORT_ERROR

    def init(self, device_id: int) -> None:
        """Initialize ACL if not already done and create/retain a device context."""
        self._ensure_acl()
        with self._lock:
            if not self._initialized:
                ret = acl.init()
                if ret != 0:
                    raise YuntuAscendOCRError(f"acl.init failed with error code {ret}")
                self._initialized = True
                atexit.register(self.finalize)

            if device_id not in self._device_contexts:
                ret = acl.rt.set_device(device_id)
                if ret != 0:
                    raise YuntuAscendOCRError(
                        f"acl.rt.set_device({device_id}) failed with error code {ret}"
                    )
                context, ret = acl.rt.create_context(device_id)
                if ret != 0:
                    raise YuntuAscendOCRError(
                        f"acl.rt.create_context({device_id}) failed with error code {ret}"
                    )
                self._device_contexts[device_id] = {
                    "context": context,
                    "refs": 1,
                }
            else:
                self._device_contexts[device_id]["refs"] += 1
            self._ref_count += 1

    def get_context(self, device_id: int):
        """Return the ACL context handle for the given device."""
        with self._lock:
            if device_id not in self._device_contexts:
                raise YuntuAscendOCRError(
                    f"ACL context for device {device_id} has not been initialized"
                )
            return self._device_contexts[device_id]["context"]

    def release(self, device_id: int) -> None:
        """Release one reference to the ACL environment."""
        self._ensure_acl()
        with self._lock:
            if device_id not in self._device_contexts:
                return
            self._device_contexts[device_id]["refs"] -= 1
            self._ref_count -= 1
            if self._device_contexts[device_id]["refs"] <= 0:
                context = self._device_contexts[device_id]["context"]
                acl.rt.destroy_context(context)
                acl.rt.reset_device(device_id)
                del self._device_contexts[device_id]

    def finalize(self) -> None:
        """Finalize ACL and clean up all remaining contexts."""
        self._ensure_acl()
        with self._lock:
            if not self._initialized:
                return
            for device_id in list(self._device_contexts.keys()):
                ctx = self._device_contexts[device_id]["context"]
                acl.rt.destroy_context(ctx)
                acl.rt.reset_device(device_id)
            self._device_contexts.clear()
            self._ref_count = 0
            acl.finalize()
            self._initialized = False
            atexit.unregister(self.finalize)


def acl_init(device_id: int) -> None:
    """Initialize ACL for the given device (idempotent)."""
    _ACLEnvironment().init(device_id)


def acl_get_context(device_id: int):
    """Get the ACL context for the given device."""
    return _ACLEnvironment().get_context(device_id)


def acl_release(device_id: int) -> None:
    """Release one ACL reference for the given device."""
    _ACLEnvironment().release(device_id)


def acl_finalize() -> None:
    """Finalize ACL globally."""
    _ACLEnvironment().finalize()
