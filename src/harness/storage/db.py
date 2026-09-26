from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

_engines: dict[str, object] = {}


def get_engine(database_url: str):
    if database_url not in _engines:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        _engines[database_url] = create_engine(database_url, connect_args=connect_args)
    return _engines[database_url]


def init_db(database_url: str) -> None:
    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)


def get_session(database_url: str) -> Session:
    return Session(get_engine(database_url))


def reset_engine(database_url: str | None = None) -> None:
    """Used by tests to force a fresh engine, e.g. between tests sharing a URL."""
    if database_url is None:
        _engines.clear()
    else:
        _engines.pop(database_url, None)
