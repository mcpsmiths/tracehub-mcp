"""The Docker opt-in recipe for FastMCP's PyPI update hint, pinned against the Dockerfile.

The image ships with FASTMCP_CHECK_FOR_UPDATES=off and FASTMCP_SHOW_SERVER_BANNER=false in its
ENV. Its default command is the HTTP transport, which never prints FastMCP's banner (the only
place the update check runs); its HEALTHCHECK probes that transport's HTTP port; and a stdio
server reads its requests from stdin. The hint itself needs the two overrides and the stdio
transport, and FastMCP prints it at startup, before it reads stdin. A stdio container as the
Dockerfile builds it also needs a stdin held open (`-i`: the server exits when stdin ends) and the
HEALTHCHECK disabled (`--no-healthcheck`, the flag docker_healthcheck.py names for stdio
deployments: a stdio server opens no port to probe). The recipe the docs once gave (both
overrides and the stdio transport) was enough for the hint but not for that container.

README.md, CLAUDE.md, PRIVACY.md and the Dockerfile's own comment each carry the recipe. This
module derives the complete `docker run` command from the Dockerfile and requires every one of
them to give it and to say which parts the hint needs, so the four cannot drift apart from each
other or from the image. Nothing here runs Docker: every expectation is read from the Dockerfile
text and the repository's code.
"""

import asyncio
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click
import pytest

from opentelemetry_mcp.server import main as server_main

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
HEALTHCHECK_SCRIPT_PATH = REPO_ROOT / "docker_healthcheck.py"

IMAGE_REPOSITORY = "ghcr.io/mcpsmiths/tracehub-mcp"

# FastMCP's own settings (not this project's). The image sets each to its "off" value; the recipe
# overrides each with the value that turns it back on.
RECIPE_OVERRIDES = {
    "FASTMCP_CHECK_FOR_UPDATES": "stable",
    "FASTMCP_SHOW_SERVER_BANNER": "true",
}


@dataclass(frozen=True)
class DockerfileFacts:
    """What the recipe depends on, read from the runtime (last) stage of a Dockerfile."""

    env: dict[str, str]
    healthcheck_command: list[str] | None
    entrypoint: list[str]
    cmd: list[str]

    @property
    def has_healthcheck(self) -> bool:
        return self.healthcheck_command is not None

    @property
    def default_transport(self) -> str:
        """The transport the image runs when no arguments follow its name. A CMD without a
        --transport flag leaves the CLI's own default (stdio, server.py's --transport option)."""
        if "--transport" not in self.cmd:
            return "stdio"
        index = self.cmd.index("--transport")
        return self.cmd[index + 1] if index + 1 < len(self.cmd) else ""


_ENV_PAIR_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')


def _exec_form(instruction: str, text: str) -> list[str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{instruction} is not in exec (JSON array) form: {text!r}") from exc
    assert isinstance(value, list), f"{instruction} must be a JSON array, got {text!r}"
    return [str(item) for item in value]


def parse_runtime_stage(dockerfile: str) -> DockerfileFacts:
    """ENV defaults, HEALTHCHECK, ENTRYPOINT and CMD of the stage after the last FROM (the
    shipped image), with backslash continuations folded."""
    runtime_stage = dockerfile.rsplit("\nFROM ", 1)[1]
    folded = re.sub(r"\\\n\s*", " ", runtime_stage)
    env: dict[str, str] = {}
    healthcheck_command: list[str] | None = None
    entrypoint: list[str] = []
    cmd: list[str] = []
    for line in folded.splitlines():
        instruction, _, rest = line.strip().partition(" ")
        if instruction == "ENV":
            for key, value in _ENV_PAIR_RE.findall(rest):
                env[key] = value.strip('"')
        elif instruction == "HEALTHCHECK" and rest.strip() != "NONE":
            healthcheck_command = _exec_form("HEALTHCHECK CMD", rest.partition(" CMD ")[2])
        elif instruction == "ENTRYPOINT":
            entrypoint = _exec_form("ENTRYPOINT", rest)
        elif instruction == "CMD":
            cmd = _exec_form("CMD", rest)
    return DockerfileFacts(
        env=env, healthcheck_command=healthcheck_command, entrypoint=entrypoint, cmd=cmd
    )


def _normalise(text: str) -> str:
    """One line of plain text: backslash continuations and Markdown backticks removed, every run
    of whitespace (including the line wraps of PRIVACY.md and the Dockerfile comment) collapsed."""
    without_continuations = re.sub(r"\\\n", " ", text)
    return re.sub(r"\s+", " ", without_continuations.replace("`", "")).strip()


# `docker run <options> <image>[:tag] <arguments>`: the arguments are the run of `--flag [value]`
# pairs right after the image, so prose that follows the command is not read as part of it.
_COMMAND_RE = re.compile(
    r"docker run (?P<options>.+?) "
    + re.escape(IMAGE_REPOSITORY)
    + r"(?::[\w.-]+)?(?P<arguments>(?: --?[\w-]+(?: [\w-]+)?)*)"
)


def recipe_problems(passage: str, facts: DockerfileFacts) -> list[str]:
    """Everything wrong with the `docker run` recipe in `passage`, judged against the Dockerfile
    `facts`. Empty means the passage gives the complete, current recipe: stdin held open, the
    HEALTHCHECK disabled when (and only when) the image has one, each override the Dockerfile ENV
    actually sets, and the stdio transport in place of a non-stdio default command."""
    command = _COMMAND_RE.search(_normalise(passage))
    if command is None:
        return [f"no `docker run ... {IMAGE_REPOSITORY} ...` command found"]

    flags: list[str] = []
    overrides: dict[str, str] = {}
    tokens = iter(command["options"].split())
    for token in tokens:
        if token in {"-e", "--env"}:
            key, _, value = next(tokens, "").partition("=")
            overrides[key] = value
        else:
            flags.append(token)

    problems: list[str] = []
    if not {"-i", "--interactive"} & set(flags):
        problems.append("missing -i: a stdio server reads its requests from stdin")
    if facts.has_healthcheck and "--no-healthcheck" not in flags:
        problems.append("missing --no-healthcheck: the image's HEALTHCHECK probes the HTTP port")
    if not facts.has_healthcheck and "--no-healthcheck" in flags:
        problems.append("stale --no-healthcheck: the Dockerfile has no HEALTHCHECK any more")
    for key, value in RECIPE_OVERRIDES.items():
        if key not in facts.env:
            if key in overrides:
                problems.append(f"stale -e {key}: the Dockerfile ENV no longer sets it")
        elif facts.env[key] == value:
            problems.append(f"-e {key}={value} overrides nothing: the Dockerfile ENV sets it too")
        elif overrides.get(key) != value:
            problems.append(f"missing -e {key}={value} (Dockerfile ENV: {facts.env[key]})")

    wanted_arguments = [] if facts.default_transport == "stdio" else ["--transport", "stdio"]
    if command["arguments"].split() != wanted_arguments:
        problems.append(
            f"arguments after the image are {command['arguments'].split()}, "
            f"want {wanted_arguments} (the image's default transport is {facts.default_transport})"
        )
    return problems


def _paragraph(text: str, marker: str) -> str:
    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.startswith(marker)]
    assert len(paragraphs) == 1, f"expected exactly one paragraph starting {marker!r}"
    return paragraphs[0]


def _markdown_section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def _comment_block(dockerfile: str, marker: str) -> str:
    """The run of `#` comment lines that begins at the line naming `marker`, markers stripped."""
    lines = dockerfile.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("#") and marker in line)
    block: list[str] = []
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        block.append(line.lstrip("#"))
    return "\n".join(block)


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


# Each document that gives the recipe, with the way to cut its recipe passage out of it. The
# markers are the stable openings of the passages; a rename shows up as a failed lookup, not as
# a vacuous pass.
RECIPE_PASSAGES: dict[str, Callable[[], str]] = {
    "README.md": lambda: _paragraph(
        _read("README.md"), "**Known third-party egress dependency (opt-in):**"
    ),
    "CLAUDE.md": lambda: _paragraph(_read("CLAUDE.md"), "**Known third-party egress (opt-in):**"),
    "PRIVACY.md": lambda: _markdown_section(
        _read("PRIVACY.md"), "## Third-party egress: FastMCP update check"
    ),
    "Dockerfile comment": lambda: _comment_block(
        _read("Dockerfile"), "FASTMCP_CHECK_FOR_UPDATES / FASTMCP_SHOW_SERVER_BANNER are"
    ),
}


@pytest.fixture(scope="module")
def facts() -> DockerfileFacts:
    return parse_runtime_stage(_read("Dockerfile"))


def test_dockerfile_still_has_the_shape_the_recipe_is_written_for(facts: DockerfileFacts) -> None:
    """Sanity check for the parser, so the tests below cannot pass vacuously, and a tripwire: the
    recipe in the four documents is written for exactly this image. A change here means every
    document's recipe needs the matching change."""
    assert facts.env["FASTMCP_CHECK_FOR_UPDATES"] == "off", facts.env
    assert facts.env["FASTMCP_SHOW_SERVER_BANNER"] == "false", facts.env
    assert facts.has_healthcheck, (
        "the image has no HEALTHCHECK: drop --no-healthcheck from the docs"
    )
    assert facts.healthcheck_command == ["python", "docker_healthcheck.py"], facts
    assert facts.entrypoint == ["tracehub-mcp"], facts.entrypoint
    assert facts.default_transport == "http", f"default CMD is {facts.cmd}: rewrite the docs"
    # Arguments after the image name replace CMD, so they are tracehub-mcp's own flags.
    assert facts.cmd[0].startswith("--"), facts.cmd


def test_healthcheck_probes_the_http_port_the_default_cmd_binds(facts: DockerfileFacts) -> None:
    """Why --no-healthcheck is part of the recipe: the HEALTHCHECK's probe targets the port the
    HTTP default command binds, and docker_healthcheck.py itself says to disable it for stdio."""
    script = HEALTHCHECK_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "http://localhost:8000/ready" in script
    assert facts.cmd[facts.cmd.index("--port") + 1] == "8000", facts.cmd
    assert "--transport stdio" in script and "--no-healthcheck" in script


def test_stdio_is_a_transport_choice_of_the_cli() -> None:
    """The recipe's `--transport stdio` must be a value the CLI accepts."""
    transport = {param.name: param for param in server_main.params}["transport"]
    assert isinstance(transport.type, click.Choice)
    assert "stdio" in transport.type.choices
    assert transport.default == "stdio"


@pytest.mark.parametrize("document", sorted(RECIPE_PASSAGES))
def test_document_gives_the_complete_recipe(document: str, facts: DockerfileFacts) -> None:
    passage = RECIPE_PASSAGES[document]()
    assert "FASTMCP_CHECK_FOR_UPDATES" in passage, f"{document}: the recipe passage was not found"
    assert recipe_problems(passage, facts) == [], f"{document}: the Docker opt-in recipe is stale"


# FastMCP prints the banner, and makes the update check inside it, before the stdio transport
# starts reading stdin (fastmcp's run_stdio_async calls log_server_banner before stdio_server()).
# So `-i` and `--no-healthcheck` are for the stdio container, not for the hint.
HINT_SCOPE_CLAIM = "-i and --no-healthcheck are not needed for the hint"


@pytest.mark.parametrize("document", sorted(RECIPE_PASSAGES))
def test_document_separates_what_the_hint_needs_from_what_a_stdio_container_needs(
    document: str,
) -> None:
    """The hint needs the two overrides and `--transport stdio`. `-i` keeps the stdio server
    running past the banner and `--no-healthcheck` avoids a probe of a port a stdio server never
    opens, so a document must not present every part as needed for the hint."""
    text = _normalise(RECIPE_PASSAGES[document]())
    assert "every part is needed" not in text.lower(), f"{document}: overstates what the hint needs"
    assert HINT_SCOPE_CLAIM in text, f"{document}: does not say which parts the hint does not need"


COMPLETE_RECIPE = (
    "docker run -i --no-healthcheck -e FASTMCP_CHECK_FOR_UPDATES=stable "
    f"-e FASTMCP_SHOW_SERVER_BANNER=true {IMAGE_REPOSITORY}:latest --transport stdio"
)


def test_checker_accepts_the_complete_recipe_however_it_is_wrapped(facts: DockerfileFacts) -> None:
    """Positive control for the checker itself: the complete command passes bare, inside
    Markdown backticks with prose around it, and split over lines with backslash continuations."""
    assert recipe_problems(COMPLETE_RECIPE, facts) == []
    assert recipe_problems(f"run it as `{COMPLETE_RECIPE}` (plus backend flags).", facts) == []
    wrapped = COMPLETE_RECIPE.replace(" -e ", " \\\n  -e ").replace(" ghcr", " \\\n  ghcr")
    assert wrapped.count("\\\n") == 3
    assert recipe_problems(wrapped, facts) == []


def test_checker_rejects_the_recipe_the_docs_gave_before(facts: DockerfileFacts) -> None:
    """The earlier recipe (both overrides and the stdio transport, nothing else) is incomplete for
    a stdio container: it does not hold stdin open and leaves the HTTP HEALTHCHECK running."""
    before = COMPLETE_RECIPE.replace(" -i", "").replace(" --no-healthcheck", "")
    assert before.startswith("docker run -e FASTMCP_CHECK_FOR_UPDATES=stable ")
    problems = recipe_problems(before, facts)
    assert [problem.split(":")[0] for problem in problems] == [
        "missing -i",
        "missing --no-healthcheck",
    ]


@pytest.mark.parametrize(
    ("dropped", "expected_problem"),
    [
        pytest.param(" -i", "missing -i", id="stdin"),
        pytest.param(" --no-healthcheck", "missing --no-healthcheck", id="healthcheck"),
        pytest.param(
            " -e FASTMCP_CHECK_FOR_UPDATES=stable",
            "missing -e FASTMCP_CHECK_FOR_UPDATES=stable",
            id="update-check-override",
        ),
        pytest.param(
            " -e FASTMCP_SHOW_SERVER_BANNER=true",
            "missing -e FASTMCP_SHOW_SERVER_BANNER=true",
            id="banner-override",
        ),
        pytest.param(" --transport stdio", "arguments after the image", id="stdio-transport"),
    ],
)
def test_checker_names_each_missing_part(
    dropped: str, expected_problem: str, facts: DockerfileFacts
) -> None:
    assert dropped in COMPLETE_RECIPE
    problems = recipe_problems(COMPLETE_RECIPE.replace(dropped, ""), facts)
    assert [problem for problem in problems if expected_problem in problem], problems


def test_checker_rejects_an_override_that_keeps_the_image_default(facts: DockerfileFacts) -> None:
    wrong = COMPLETE_RECIPE.replace("SHOW_SERVER_BANNER=true", "SHOW_SERVER_BANNER=false")
    assert [p for p in recipe_problems(wrong, facts) if "missing -e FASTMCP_SHOW" in p]


def _without_instruction(dockerfile: str, instruction: str) -> str:
    return re.sub(rf"^{instruction}(?:[^\n]*\\\n)*[^\n]*\n", "", dockerfile, flags=re.MULTILINE)


@pytest.mark.parametrize(
    ("variant", "expected_problem"),
    [
        pytest.param(
            lambda text: _without_instruction(text, "HEALTHCHECK"),
            "stale --no-healthcheck",
            id="healthcheck-removed",
        ),
        pytest.param(
            lambda text: re.sub(r"^CMD .*$", 'CMD ["--transport", "stdio"]', text, flags=re.M),
            "arguments after the image",
            id="default-cmd-is-stdio",
        ),
        pytest.param(
            lambda text: text.replace(
                'FASTMCP_CHECK_FOR_UPDATES="off"', 'FASTMCP_CHECK_FOR_UPDATES="stable"'
            ),
            "overrides nothing",
            id="env-already-opted-in",
        ),
        pytest.param(
            lambda text: re.sub(r' \\\n\s*FASTMCP_SHOW_SERVER_BANNER="false"', "", text),
            "stale -e FASTMCP_SHOW_SERVER_BANNER",
            id="banner-env-removed",
        ),
    ],
)
def test_a_changed_dockerfile_makes_every_document_stale(
    variant: Callable[[str], str], expected_problem: str
) -> None:
    """The recipe is derived from the Dockerfile, not frozen: change the image and the unchanged
    documents are reported stale (here, against each document as it stands)."""
    original = _read("Dockerfile")
    changed = variant(original)
    assert changed != original, "the variant did not change the Dockerfile: fix the test"
    changed_facts = parse_runtime_stage(changed)
    for document, passage in RECIPE_PASSAGES.items():
        problems = recipe_problems(passage(), changed_facts)
        assert any(expected_problem in problem for problem in problems), (document, problems)


_SKIPPED_DIRECTORIES = {".git", ".venv", "node_modules", ".mypy_cache", ".ruff_cache", "tests"}
_TEXT_SUFFIXES = {".md", ".yml", ".yaml", ".json", ".toml", ".sh", ".txt", ".example"}
REGISTERED_DOCUMENTS = {"README.md", "CLAUDE.md", "PRIVACY.md", "Dockerfile"}
OVERRIDE_NEEDLE = "-e FASTMCP_CHECK_FOR_UPDATES=stable"


async def _git_output(root: Path, *arguments: str) -> bytes | None:
    """stdout of `git -C <root> <arguments>`, or None when git is missing or the command fails.
    GIT_* variables are dropped so a surrounding hook's GIT_DIR or GIT_INDEX_FILE cannot redirect
    the command to another repository."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
        )
    except OSError:
        return None
    stdout, _ = await process.communicate()
    return stdout if process.returncode == 0 else None


async def _checkout_files(root: Path) -> list[Path] | None:
    """The files of the git checkout whose top directory is `root`: tracked files plus new files
    git does not ignore. None when `root` is not the top of a git work tree, or git is missing."""
    top = await _git_output(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.decode("utf-8").strip()).resolve() != root.resolve():
        return None
    listing = await _git_output(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    )
    if listing is None:
        return None
    return [root / name for name in listing.decode("utf-8").split("\0") if name]


def _walked_files(root: Path) -> list[Path]:
    """Every file under `root` outside the skipped directories (the fallback when git cannot say
    what belongs to the repository, e.g. an unpacked source archive)."""
    files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name not in _SKIPPED_DIRECTORIES]
        files.extend(Path(directory, filename) for filename in filenames)
    return files


def _is_searched(path: Path, root: Path) -> bool:
    """A text file outside the skipped directories. CHANGELOG.md is generated history, not
    guidance, so it is not searched."""
    if _SKIPPED_DIRECTORIES & set(path.relative_to(root).parts[:-1]):
        return False
    if path.name != "Dockerfile" and path.suffix not in _TEXT_SUFFIXES:
        return False
    return path.name != "CHANGELOG.md" and path.is_file()


async def _files_carrying_the_override(root: Path) -> set[str]:
    """Paths (relative to `root`, POSIX style) of every searched file of the repository that
    gives the override. The repository is what git lists; gitignored scratch notes, build output
    and local evidence are not its documents, and a clean CI checkout never has them."""
    candidates = await _checkout_files(root)
    if candidates is None:
        candidates = _walked_files(root)
    return {
        path.relative_to(root).as_posix()
        for path in candidates
        if _is_searched(path, root)
        and OVERRIDE_NEEDLE in _normalise(path.read_text(encoding="utf-8", errors="ignore"))
    }


async def test_no_other_file_carries_a_copy_of_the_recipe() -> None:
    """A fifth copy would drift unnoticed: any file that tells operators to pass the override
    must be one of the documents checked above."""
    carrying = await _files_carrying_the_override(REPO_ROOT)
    assert REGISTERED_DOCUMENTS <= carrying, "the search missed a registered document"
    assert carrying - REGISTERED_DOCUMENTS == set(), "add these documents to RECIPE_PASSAGES"


_COPY_OF_THE_RECIPE = f"Run it as {COMPLETE_RECIPE} (plus your backend flags).\n"


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def test_only_the_repositorys_own_files_count_as_carrying_a_copy(tmp_path: Path) -> None:
    """The documents are the files of the repository: tracked ones, and new ones git does not
    ignore. A gitignored directory (scratch notes, build output, local evidence) is not the
    documentation and a clean CI checkout never has it, so a copy there must not turn a local
    `pytest` run red. A tracked file deleted from the work tree must not break the search."""
    assert await _git_output(tmp_path, "init", "--quiet") is not None, "git init failed"
    _write(tmp_path, ".gitignore", "scratch-pad/\ndist/\n")
    _write(tmp_path, "README.md", "No recipe here.\n")
    _write(tmp_path, "docs/tracked.md", _COPY_OF_THE_RECIPE)
    _write(tmp_path, "docs/deleted.md", _COPY_OF_THE_RECIPE)
    _write(tmp_path, "docs/new-and-not-staged.md", _COPY_OF_THE_RECIPE)
    _write(tmp_path, "scratch-pad/evidence/run.txt", _COPY_OF_THE_RECIPE)
    _write(tmp_path, "dist/readme-copy.md", _COPY_OF_THE_RECIPE)
    staged = await _git_output(
        tmp_path, "add", ".gitignore", "README.md", "docs/tracked.md", "docs/deleted.md"
    )
    assert staged is not None, "git add failed"
    (tmp_path / "docs" / "deleted.md").unlink()

    assert await _files_carrying_the_override(tmp_path) == {
        "docs/tracked.md",
        "docs/new-and-not-staged.md",
    }


async def test_outside_a_git_checkout_every_searched_text_file_counts(tmp_path: Path) -> None:
    """Without git to say what belongs to the repository (an unpacked source archive), the search
    falls back to walking the tree: all searched text files except the skipped directories and
    the generated CHANGELOG.md."""
    _write(tmp_path, "docs/guide.md", _COPY_OF_THE_RECIPE)
    _write(tmp_path, ".venv/lib/readme.md", _COPY_OF_THE_RECIPE)
    _write(tmp_path, "CHANGELOG.md", _COPY_OF_THE_RECIPE)
    _write(tmp_path, "main.py", _COPY_OF_THE_RECIPE)

    assert await _files_carrying_the_override(tmp_path) == {"docs/guide.md"}
