from __future__ import annotations

import logging
import sys
import textwrap
import traceback

_CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"  # traceback's own words
_CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n\n"


class WithoutMessages(logging.Filter):
    """A logged exception (`log.exception`, uvicorn's "Exception in ASGI application") by its types and frames alone
    (`traceback_without_messages`): in the processes that handle requests, an exception's message can quote one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = traceback_without_messages(record.exc_info[1]).rstrip()
            record.exc_info = None
        return True


def setup_logging(level: str, process: str, without_messages: bool = False) -> None:
    """One format for every process: `INFO 10-03 14:00:00.123 [engine] message`. `without_messages`: logged
    exceptions without their messages (`WithoutMessages`), for the processes that handle requests' contents."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(f"%(levelname)s %(asctime)s.%(msecs)03d [{process}] %(message)s",
                                           datefmt="%m-%d %H:%M:%S"))
    if without_messages:
        handler.addFilter(WithoutMessages())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # huggingface_hub's HTTP client (httpx2) logs every request of a download at INFO (llm_mini_infer.hub logs the
    # download)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx2", "httpcore2"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def traceback_without_messages(exc: BaseException) -> str:
    """`exc`'s traceback as Python prints it (after those of the exceptions it was raised from or while handling; an
    exception group's members, numbered, after the group's own), but each exception named by its type alone: no message
    and no notes, which can quote what a request carried (a prompt, a schema, an output)."""
    chain: list[BaseException] = []
    e: BaseException | None = exc
    while e is not None and not any(e is seen for seen in chain):
        chain.append(e)
        e = e.__cause__ if e.__cause__ is not None else None if e.__suppress_context__ else e.__context__
    out = []
    for i, e in enumerate(reversed(chain)):
        if i:
            out.append(_CAUSE if e.__cause__ is not None else _CONTEXT)
        if e.__traceback__ is not None:
            out.append("Traceback (most recent call last):\n")
            out.extend(traceback.format_tb(e.__traceback__))
        t = type(e)
        out.append((t.__qualname__ if t.__module__ in ("builtins", "__main__") else f"{t.__module__}.{t.__qualname__}")
                   + "\n")
        if isinstance(e, BaseExceptionGroup):
            for j, member in enumerate(e.exceptions, 1):
                out.append(f"+---------------- {j} ----------------\n")
                out.append(textwrap.indent(traceback_without_messages(member), "| "))
            out.append("+------------------------------------\n")
    return "".join(out)
