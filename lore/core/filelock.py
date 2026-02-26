"""
OS-specific file locking
"""
import time

HAS_FCNTL = None
HAS_MSVCRT = None
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False

def acquire_lock(file_obj, timeout=10):
    """Dependency-free cross-platform file locking."""
    start_time = time.time()
    while True:
        try:
            if HAS_FCNTL:
                # LOCK_EX = Exclusive, LOCK_NB = Non-blocking
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            elif HAS_MSVCRT:
                # LK_NBLCK = Non-blocking lock, lock the first byte
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                return
            else:
                # Fallback if somehow on an OS with neither
                return
        except OSError:
            if time.time() - start_time >= timeout:
                raise RuntimeError("Timeout acquiring session lock.")
            time.sleep(0.1) # Wait 100ms and try again

def release_lock(file_obj):
    """Release the file lock."""
    if HAS_FCNTL:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    elif HAS_MSVCRT:
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
