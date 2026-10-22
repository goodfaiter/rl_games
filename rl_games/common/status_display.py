import shutil
import sys
import time


class StatusDisplay:
    """Training status kept at the bottom of the terminal, with events printed above it.

    In a terminal, ``update(lines)`` redraws the status lines in place (ANSI cursor moves, no scrolling) and
    ``event(message)`` prints a line above them that stays in the history. Other prints (to sys.stdout) also
    land above the status, which is redrawn under them. When stdout is not a terminal (e.g. redirected to a log
    file) every ``update`` writes its lines once, joined into one plain line.
    """

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.live = self.stream.isatty()
        self.lines = []
        self.drawn = 0
        if self.live and stream is None:
            sys.stdout = _StdoutAbove(self)

    def _clear(self):
        # Move up over the drawn status lines and clear them (the cursor ends where the first one was).
        for _ in range(self.drawn):
            self.stream.write('\x1b[1A\x1b[2K')
        self.drawn = 0

    def _draw(self):
        width = shutil.get_terminal_size((120, 20)).columns
        for line in self.lines:
            self.stream.write((line if len(line) < width else line[:width - 2] + '…') + '\n')
        self.drawn = len(self.lines)
        self.stream.flush()

    def update(self, lines):
        self.lines = list(lines)
        if self.live:
            self._clear()
            self._draw()
        else:
            self.stream.write(' | '.join(self.lines) + '\n')
            self.stream.flush()

    def event(self, message):
        if self.live:
            self._clear()
            self.stream.write(f'[{time.strftime("%H:%M:%S")}] {message}\n')
            self._draw()
        else:
            self.stream.write(message + '\n')
            self.stream.flush()


class _StdoutAbove:
    """sys.stdout while the status is live: other prints clear the status, go above it, and it is redrawn."""

    def __init__(self, display):
        self.display = display
        self.stream = display.stream

    def write(self, text):
        if self.display.drawn:
            self.display._clear()
        self.stream.write(text)
        if text.endswith('\n') and self.display.lines:
            self.display._draw()
        return len(text)

    def __getattr__(self, name):  # flush, isatty, fileno, encoding, ...
        return getattr(self.stream, name)


def progress_bar(fraction, width=24):
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return '[' + '█' * filled + '░' * (width - filled) + ']'


def compact(value):
    """1234567 -> '1.2M', 156000 -> '156k'."""
    for factor, suffix in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(value) >= factor:
            return f'{value / factor:.1f}{suffix}' if abs(value) < 10 * factor else f'{value / factor:.0f}{suffix}'
    return f'{value:.0f}'


def duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes}:{seconds:02d}'
