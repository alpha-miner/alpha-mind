import pytest
import sqlalchemy as sa

from alphamind.bin.cli import get_parser, main
from alphamind.data.dbmodel import models


def test_cli_parser_requires_subcommand():
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["initdb", "--url", "sqlite:///:memory:"])
    assert args.subcommand == "initdb"
    assert args.url == "sqlite:///:memory:"


def test_initdb_creates_complete_schema(tmp_path):
    url = "sqlite+pysqlite:///" + str(tmp_path / "cli.sqlite")
    main(["initdb", "--url", url])

    inspector = sa.inspect(sa.create_engine(url))
    tables = set(inspector.get_table_names())
    assert models.Market.__tablename__ in tables
    assert models.FactorMaster.__tablename__ in tables
    assert models.RiskExposure.__tablename__ in tables
