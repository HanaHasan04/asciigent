"""The .env loader must never clobber a real environment variable."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.env import find_dotenv, load_env, parse_dotenv  # noqa: E402


def test_parses_the_shapes_people_actually_write():
    parsed = parse_dotenv(
        "\n".join(
            [
                "# a comment",
                "",
                "GEMINI_API_KEY=abc123",
                'QUOTED="with spaces"',
                "SINGLE='sq'",
                "export EXPORTED=yes",
                "  SPACED = padded  ",
                "novalue",
            ]
        )
    )
    assert parsed["GEMINI_API_KEY"] == "abc123"
    assert parsed["QUOTED"] == "with spaces"
    assert parsed["SINGLE"] == "sq"
    assert parsed["EXPORTED"] == "yes"
    assert parsed["SPACED"] == "padded"
    assert "novalue" not in parsed


def test_keys_with_equals_signs_survive():
    assert parse_dotenv("K=a=b=c")["K"] == "a=b=c"


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("MY_TEST_KEY", "from_shell")
    load_env(env_file)
    assert os.environ["MY_TEST_KEY"] == "from_shell"


def test_override_is_available_when_asked(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("MY_TEST_KEY", "from_shell")
    load_env(env_file, override=True)
    assert os.environ["MY_TEST_KEY"] == "from_file"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_find_walks_up_to_the_repo_root(tmp_path):
    (tmp_path / ".env").write_text("A=1", encoding="utf-8")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_dotenv(deep / "file.py") == tmp_path / ".env"
