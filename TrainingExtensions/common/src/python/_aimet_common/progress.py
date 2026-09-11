# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Progress bar utility shared across optimization loops (e.g. AdaScale)."""

import sys

# tqdm.auto resolves to tqdm.notebook (an ipywidgets progress bar, rendered in
# its own output area so interleaved log lines cannot break it) when running
# under an IPython kernel with ipywidgets installed, and to the console tqdm
# everywhere else. ipywidgets is not an AIMET dependency, so the console
# fallback below still has to stay readable in a notebook.
from tqdm.auto import tqdm
from tqdm.notebook import tqdm_notebook

# True when tqdm.auto gave us the ipywidgets bar. A widget updates in place no
# matter how often it is refreshed, so it keeps tqdm's default refresh rate.
_IS_WIDGET = issubclass(tqdm, tqdm_notebook)


def _renders_in_place(file) -> bool:
    """Return True if the bar can be redrawn without appending to a log."""
    if _IS_WIDGET:
        return True
    # tqdm writes to sys.stderr unless the caller passes an explicit stream.
    stream = file if file is not None else sys.stderr
    try:
        return bool(stream.isatty())
    except Exception:  # pylint: disable=broad-except
        return False


# tqdm.auto.tqdm carries the same disable: its own bases are only consistent
# once the environment picks one of them.
class _SummaryOnlyTqdm(tqdm):  # pylint: disable=inconsistent-mro
    """
    A :class:`tqdm` which draws nothing while it runs and prints its usual
    status line once, when it is closed.

    A terminal overwrites the previous line, so a progress bar can be redrawn
    as often as it likes. Anything else (a redirected file, a pipe, a notebook
    without ipywidgets) captures the output line by line and turns every redraw
    into a separate log message. Suppressing every intermediate redraw leaves
    exactly one line per bar - the same summary (count, elapsed, rate) tqdm
    would have left behind on completion.
    """

    # Guards the redraws. Set before ``tqdm.__init__``, which displays the bar
    # in its initial state, and as a class attribute so that ``__del__`` on a
    # half-constructed instance still finds it.
    _drawing = False

    def display(self, *args, **kwargs) -> bool:
        if not self._drawing:
            return False
        return super().display(*args, **kwargs)

    def _write_summary(self, msg):
        """Write one plain line, in place of tqdm's line-overwriting printer."""
        self.fp.write(str(msg))
        getattr(self.fp, "flush", lambda: None)()

    def close(self):
        if self.disable:
            return
        # Even a nested bar (leave=False, normally erased on close) has a
        # summary worth keeping: without redraws it is the only record that the
        # block ran at all. tqdm.close() renders the final line via display().
        self.leave = True
        # tqdm.close() skips the final line for a bar that never displayed,
        # which is every bar here; that check exists to avoid clearing a line
        # a delayed bar never drew, and there is nothing to clear either way.
        self.delay = 0
        # tqdm's printer prefixes '\r' to overwrite the line it drew last, which
        # a log file records as a stray control character. Nothing was drawn.
        self.sp = self._write_summary
        self._drawing = True
        try:
            super().close()
        finally:
            self._drawing = False


def progress_bar(*args, **kwargs) -> tqdm:
    """
    Create a :class:`tqdm` progress bar which animates in a terminal or as a
    notebook widget, and degrades to a single end-of-run summary line wherever
    the output is captured line by line (a pipe, a log file, a notebook
    without ipywidgets).

    Accepts the same arguments as :class:`tqdm`.
    """
    bar_type = tqdm if _renders_in_place(kwargs.get("file")) else _SummaryOnlyTqdm
    return bar_type(*args, **kwargs)
