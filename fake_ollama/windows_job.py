"""Windows process-tree identity retained even after the launcher exits.

The child starts suspended, is assigned to a non-breakaway Job Object, then
its primary thread resumes. No PID/name-based tree termination is used.
"""
import ctypes
from ctypes import wintypes as w


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", w.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                ("active", w.DWORD), ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class IOCounts(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IOCounts), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]


class Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
        (name, w.DWORD) for name in ("faults", "total", "active", "terminated")]


class ThreadEntry(ctypes.Structure):
    _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("tid", w.DWORD), ("pid", w.DWORD),
                ("base", w.LONG), ("delta", w.LONG), ("flags", w.DWORD)]


class WindowsJob:
    def __init__(self):
        self._retained_process_handles = {}
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
            "Thread32Next": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.kernel, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        self._check(self.handle)
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self._check(self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_and_resume(self, pid):
        process = self.kernel.OpenProcess(0x100 | 0x1, False, pid)  # SET_QUOTA | TERMINATE
        self._check(process)
        try:
            self._check(self.kernel.AssignProcessToJobObject(self.handle, process))
        finally:
            self.kernel.CloseHandle(process)
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x4, 0)  # THREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.pid == pid:
                    thread = self.kernel.OpenThread(0x2, False, entry.tid)
                    self._check(thread)
                    try:
                        if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        self.kernel.CloseHandle(thread)
                    return
                found = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError("Suspended process primary thread was not found")
        finally:
            self.kernel.CloseHandle(snapshot)

    def active(self):
        if not self.handle:
            return 0
        info = Accounting()
        self._check(self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.active

    def terminate(self):
        self._check(self.kernel.TerminateJobObject(self.handle, 1))

    def process_handles(self):
        if not self.handle:
            return list(self._retained_process_handles.values())
        for pid, handle in list(self._retained_process_handles.items()):
            if self.kernel.WaitForSingleObject(handle, 0) == 0:
                self.kernel.CloseHandle(handle)
                del self._retained_process_handles[pid]
        capacity = 16
        while True:
            data = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            if self.kernel.QueryInformationJobObject(self.handle, 3, data, ctypes.sizeof(data), None):
                break
            if ctypes.get_last_error() != 234 or capacity > 65536:
                raise ctypes.WinError(ctypes.get_last_error())
            capacity *= 2
        count = w.DWORD.from_buffer(data, 4).value
        pids = (ctypes.c_size_t * count).from_buffer(data, 8)
        handles = {}
        try:
            for pid in pids:
                if pid in self._retained_process_handles:
                    continue
                handle = self.kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
                if handle:
                    handles[pid] = handle
                elif ctypes.get_last_error() != 87:  # already exited
                    raise ctypes.WinError(ctypes.get_last_error())
            self._retained_process_handles.update(handles)
            return list(self._retained_process_handles.values())
        except BaseException:
            for handle in handles.values():
                self.kernel.CloseHandle(handle)
            raise

    def running(self):
        return bool(self.active() or any(self.kernel.WaitForSingleObject(handle, 0) != 0
                    for handle in self._retained_process_handles.values()))

    def close(self):
        for handle in getattr(self, "_retained_process_handles", {}).values():
            self.kernel.CloseHandle(handle)
        self._retained_process_handles = {}
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
