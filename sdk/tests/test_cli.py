from click.testing import CliRunner
from atriumdb.cli.hello import hello


def test_cli():
    runner = CliRunner()
    result = runner.invoke(hello)
    print()
    print(result.output)
