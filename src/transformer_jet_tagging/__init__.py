import pathlib
import subprocess

from ._version import __version__ as __base_version__


def _git_suffix() -> str:
    """
    Add the necessary information to the version string.
    """
    # pylint: disable=broad-except
    kwargs = dict(cwd=pathlib.Path(__file__).parent, stderr=subprocess.DEVNULL)
    try:
        # Retrieve the git short sha to be appended to the base version string.
        args = ["git", "rev-parse", "--short", "HEAD"]
        sha = subprocess.check_output(args, **kwargs).decode().strip()
        suffix = f"+g{sha}"
        # If we have uncommitted changes, append a `.dirty` to the version suffix.
        args = ["git", "diff", "--quiet"]
        if subprocess.call(args, stdout=subprocess.DEVNULL, **kwargs) != 0:
            suffix = f"{suffix}.dirty"
        return suffix
    except Exception:
        return ""


__version__ = f"{__base_version__}{_git_suffix()}"
