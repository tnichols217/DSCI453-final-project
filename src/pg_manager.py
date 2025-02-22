"""Models for using SQLAlchemy"""

from typing import Any, Literal

from sqlalchemy import (
    BigInteger,
    Column,
    Integer,
    MetaData,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import ARRAY


class BaseType:
    """Base class for SQLAlchemy models."""

    def __init__(**kwargs: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]  # noqa: ANN401
        """Initialize the base class."""
        super().__init__(**kwargs)

    metadata: type = MetaData


Base: type[BaseType] = declarative_base(metadata=MetaData(schema="public"))
metadata = Base.metadata


class Image(Base):
    """Image model."""

    __tablename__: Literal["images"] = "images"

    id: Column[int] = Column(BigInteger, primary_key=True, nullable=False)
    red: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    gre: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    blu: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    hue: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    sat: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    val: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    edge: Column[list[list[int]]] = Column(ARRAY(Integer, dimensions=2), nullable=False)
    # dilate  integer[500][500] not null,
    # erode   integer[500][500] not null,
