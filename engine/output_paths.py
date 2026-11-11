"""Shared output naming and no-overwrite publication for offline dubbing."""

import errno
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def mp4_name(source, *, lip_sync=False):
    stem = Path(source).stem
    return f"{stem}{'_lipsync' if lip_sync else ''}.mp4"


def available_mp4_path(directory, source, *, lip_sync=False):
    destination = Path(directory) / mp4_name(source, lip_sync=lip_sync)
    candidate = destination
    number = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = destination.with_name(f"{destination.stem}-{number}.mp4")
        number += 1
    return candidate


def checked_mp4_destination(path):
    destination = Path(path).expanduser().absolute()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Dubbed output must use an .mp4 filename (H.264 video and AAC audio).")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}. Choose another .mp4 filename.")
    return destination


@contextmanager
def publish_mp4(path):
    """Render beside the destination and publish the completed file without clobbering it."""
    destination = checked_mp4_destination(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".voxadub-", dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        yield staged
        if not staged.is_file() or not staged.stat().st_size:
            raise RuntimeError("The dubbing process did not create a usable MP4.")
        try:
            os.link(staged, destination)
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}:
                raise
            # Filesystems without hard links still support exclusive creation.
            with destination.open("xb"):
                pass
            try:
                os.replace(staged, destination)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
