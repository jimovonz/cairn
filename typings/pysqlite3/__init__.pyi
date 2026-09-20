"""Type stub for pysqlite3.

pysqlite3 is a C extension (a newer SQLite than the stdlib's) with no shipped
stubs, so a checker reports `pysqlite3.Connection`, `.connect`, `.OperationalError`
etc. as unknown — ~55 findings across the cairn modules that carry the sqlite
guard. Its API mirrors the stdlib `sqlite3` module exactly, so re-export the
stdlib types here. This is types-only: runtime still imports pysqlite3 (see the
guard at the top of each module), which is the whole point of the guard.

Lives in ./typings, pyright's default stubPath, so no config is needed.
"""
from sqlite3 import *  # noqa: F401,F403
from sqlite3 import (
    Connection as Connection,
    Cursor as Cursor,
    Error as Error,
    DatabaseError as DatabaseError,
    IntegrityError as IntegrityError,
    InterfaceError as InterfaceError,
    InternalError as InternalError,
    NotSupportedError as NotSupportedError,
    OperationalError as OperationalError,
    ProgrammingError as ProgrammingError,
    Warning as Warning,
    Row as Row,
    connect as connect,
    complete_statement as complete_statement,
    register_adapter as register_adapter,
    register_converter as register_converter,
    enable_callback_tracebacks as enable_callback_tracebacks,
)
