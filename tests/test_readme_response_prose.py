"""README "What responses report and what they do not": prose sentences read back and tied to behaviour.

Two sentences carry claims that no test used to read: the opening one, that a response has no field
naming the backend, the applied limit, a redaction policy or truncation (and the Redaction row's
"No policy field"), and the Failed hydration one, that Datadog, Sentry, New Relic, AWS X-Ray and
Honeycomb return count 0 when every hydration fails while Tempo raises. The opening sentence is tied
to the field names the schema-and-response test proves absent from every tool; the Failed hydration
sentence is parsed, and each backend it names is then run through the real tool and must do what the
sentence says it does.
"""

import re
from collections.abc import Awaitable, Callable, Iterator
from unittest.mock import AsyncMock, patch

import pytest

import opentelemetry_mcp.server as server
from tests.readme_response_batch_support import (
    HoneycombLoopback,
    XRayLoopback,
    honeycomb_ids,
    xray_ids,
)
from tests.readme_response_support import (
    _SAYS_SO,
    _body,
    _call,
    _install,
    _row,
    _section,
    _TempoFake,
)
from tests.test_readme_response_limits import _hydrating_backend
from tests.test_readme_response_windows_and_redaction import _ABSENT_FIELD_NAMES

_OPENING = (
    "They have no field that names the backend, states the `limit` the server applied or states "
    "a redaction policy, and no general flag for truncation"
)
# What each promise in that sentence means as a response or schema field name.
_PROMISES = {
    "names the backend": {"backend"},
    "states the `limit` the server applied": {"limit", "requested_limit", "applied_limit"},
    "states a redaction policy": {"redaction", "redaction_policy"},
    "no general flag for truncation": {"truncated", "is_truncated", "partial", "is_partial"},
}
_COUNT_ZERO_CLAUSE = re.compile(
    r"^No\. If every fetch fails, (?P<first>.+?) return `count: 0`, the same as no matches, "
    r"and so do (?P<rest>.+?)\. (?P<raises>[A-Z][A-Za-z ]+?) raises an error when"
)


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


# -- the opening sentence and the Redaction row: no provenance or policy field -------------


def test_the_opening_sentence_promises_no_backend_limit_policy_or_truncation_field() -> None:
    opening = _section().split("|", 1)[0]

    assert _OPENING in opening
    assert opening.count("They have no field") == 1
    for promise, field_names in _PROMISES.items():
        assert promise in _OPENING, promise
        assert field_names <= _ABSENT_FIELD_NAMES, (promise, field_names - _ABSENT_FIELD_NAMES)


def test_the_redaction_row_says_there_is_no_policy_field() -> None:
    assert _row("Redaction")[_SAYS_SO] == "No policy field"
    assert {"redaction", "redaction_policy"} <= _ABSENT_FIELD_NAMES


# -- the Failed hydration sentence: who returns count 0 and who raises ----------------------


def _names(text: str) -> list[str]:
    """'Datadog, Sentry and New Relic' -> three names; parentheticals are dropped."""
    plain = re.sub(r"\s*\([^)]*\)", "", text)
    return [name.strip() for name in re.split(r",| and ", plain) if name.strip()]


def _readme_groups() -> tuple[list[str], list[str]]:
    """(backends the README says return count 0, backends it says raise) when every fetch fails."""
    says = _row("Failed hydration")[_SAYS_SO]
    match = _COUNT_ZERO_CLAUSE.search(says)
    assert match, "the Failed hydration cell no longer reads 'If every fetch fails, <A> return ...'"
    return [*_names(match["first"]), *_names(match["rest"])], [match["raises"]]


async def _outcome() -> tuple[bool, int | None]:
    """(search raised, count) for a search_traces call through the real server tool."""
    result = await _call("search_traces", {})
    return result.is_error, None if result.is_error else _body(result)["count"]


def _count_zero_scenario(kind: str) -> Callable[[], Awaitable[tuple[bool, int | None]]]:
    async def scenario() -> tuple[bool, int | None]:
        backend, discovery, rows = _hydrating_backend(kind)
        with (
            patch.object(backend, discovery, AsyncMock(return_value=rows)),
            patch.object(backend, "get_trace", AsyncMock(side_effect=RuntimeError("outage"))),
        ):
            await _install(backend)
            return await _outcome()

    return scenario


async def _honeycomb() -> tuple[bool, int | None]:
    loopback = HoneycombLoopback(honeycomb_ids(2), empty=True)
    with loopback.patched():
        await _install(loopback.backend)
        return await _outcome()


async def _xray() -> tuple[bool, int | None]:
    loopback = XRayLoopback(xray_ids(2), unprocessed=True)
    with loopback.patched():
        await _install(loopback.backend)
        return await _outcome()


async def _tempo() -> tuple[bool, int | None]:
    _TempoFake(candidates=2, all_fail=True).install()
    return await _outcome()


_EVERY_FETCH_FAILS = {
    "Datadog": _count_zero_scenario("datadog"),
    "Sentry": _count_zero_scenario("sentry"),
    "New Relic": _count_zero_scenario("newrelic"),
    "AWS X-Ray": _xray,
    "Honeycomb": _honeycomb,
    "Tempo": _tempo,
}


def test_the_sentence_names_every_hydrating_backend_exactly_once() -> None:
    zero, raises = _readme_groups()

    assert sorted([*zero, *raises]) == sorted(_EVERY_FETCH_FAILS)


@pytest.mark.parametrize("name", sorted(_EVERY_FETCH_FAILS))
async def test_each_backend_does_what_the_failed_hydration_sentence_says_when_every_fetch_fails(
    name: str,
) -> None:
    zero, raises = _readme_groups()

    raised, count = await _EVERY_FETCH_FAILS[name]()

    if name in zero:
        assert (raised, count) == (False, 0), name
    else:
        assert name in raises, f"{name} is in neither group of the README sentence"
        assert raised, name
