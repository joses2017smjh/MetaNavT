"""Regular-file operations anchored to non-symlink directory descriptors.

Mutations are serialized by FilesystemTools. This protects the tool boundary;
it does not isolate an adversarial process with the same OS account.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import stat


def relative(root: Path, path: str) -> str:
    value = Path(path)
    if value.is_absolute():
        try:
            value = value.relative_to(root)
        except ValueError as exc:
            raise PermissionError("path escapes corpus root") from exc
    if ".." in value.parts or not value.parts:
        raise PermissionError("a nonempty path within the corpus is required")
    return str(value)


@contextmanager
def parent_fd(root: Path, path: str, *, create: bool = False):
    parts = Path(relative(root, path)).parts
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd, parts[-1]
    finally:
        os.close(fd)


def fingerprint(root: Path, path: str) -> dict:
    """Bind content, file identity and existing parent directory identities."""
    path = relative(root, path)
    current = root
    parents = []
    for part in Path(path).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise PermissionError("symlink component refused")
        try:
            st = current.stat()
        except FileNotFoundError:
            return {"path": path, "exists": False, "parents": parents}
        if not stat.S_ISDIR(st.st_mode):
            raise PermissionError("parent is not a directory")
        parents.append([str(current.relative_to(root)), st.st_dev, st.st_ino])
    try:
        with parent_fd(root, path) as (directory, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode):
                    raise PermissionError("only regular files may be changed")
                digest = hashlib.sha256()
                while data := os.read(fd, 1 << 20):
                    digest.update(data)
                after = os.fstat(fd)
                if (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise PermissionError("file changed while preparing review")
                return {"path": path, "exists": True, "sha256": digest.hexdigest(),
                        "dev": st.st_dev, "inode": st.st_ino, "size": st.st_size,
                        "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
                        "mode": st.st_mode, "parents": parents}
            finally:
                os.close(fd)
    except FileNotFoundError:
        return {"path": path, "exists": False, "parents": parents}


def create_file(root: Path, path: str, data: bytes) -> None:
    with parent_fd(root, path, create=True) as (directory, name):
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)


def replace_file(root: Path, path: str, data: bytes, expected: dict) -> None:
    with parent_fd(root, path) as (directory, name):
        temporary = ".metanavit-" + secrets.token_hex(16)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
            os.fchmod(fd, stat.S_IMODE(expected["mode"]))
            if fingerprint(root, path) != expected:
                raise PermissionError("reviewed target changed before replacement")
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            os.close(fd)
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass


def move_file(root: Path, src: str, dst: str, expected: dict) -> None:
    """No-overwrite move for regular files on the same filesystem."""
    with parent_fd(root, src) as (src_fd, src_name), parent_fd(root, dst, create=True) as (dst_fd, dst_name):
        os.link(src_name, dst_name, src_dir_fd=src_fd, dst_dir_fd=dst_fd, follow_symlinks=False)
        try:
            st = os.stat(dst_name, dir_fd=dst_fd, follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (expected["dev"], expected["inode"]):
                raise PermissionError("reviewed source changed before move")
            source = os.stat(src_name, dir_fd=src_fd, follow_symlinks=False)
            if (source.st_dev, source.st_ino) != (st.st_dev, st.st_ino):
                raise PermissionError("reviewed source was replaced before move")
            os.unlink(src_name, dir_fd=src_fd)
        except BaseException:
            os.unlink(dst_name, dir_fd=dst_fd)
            raise
