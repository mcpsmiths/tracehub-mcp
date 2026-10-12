"""Tests for the host-classification and userinfo-redaction helpers in
security.py."""

import io
import logging
import sys

import pytest
from pydantic import HttpUrl, ValidationError

from opentelemetry_mcp.config import BackendConfig
from opentelemetry_mcp.security import (
    UserinfoRedactingFilter,
    is_cloud_metadata_host,
    is_loopback_host,
    redact_url_userinfo,
    redact_userinfo_in_text,
)

# Dummy marker standing in for a credential; nothing real.
VFY_MARKER = "VFY-PW-5a9d"


class TestIsCloudMetadataHost:
    @pytest.mark.parametrize(
        "host",
        [
            "169.254.169.254",
            "169.254.170.2",
            "fd00:ec2::254",
            "[fd00:ec2::254]",
            "100.100.100.200",
            "metadata.google.internal",
            "METADATA.GOOGLE.INTERNAL.",
            "::ffff:169.254.169.254",
        ],
    )
    def test_canonical_spellings_are_blocked(self, host: str) -> None:
        assert is_cloud_metadata_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "2852039166",  # decimal
            "0xa9fea9fe",  # hex
            "169.254.43518",  # short form (3 parts)
            "0xa9.0xfe.0xa9.0xfe",  # hex-dotted
            "0251.0376.0251.0376",  # octal
            "168.63.129.16",  # Azure WireServer
        ],
    )
    def test_encoded_spellings_and_unlisted_metadata_hosts_are_blocked(self, host: str) -> None:
        """httpx keeps these as plain hostnames and libc's resolver maps them
        to 169.254.169.254 - a backend-chosen redirect hop must be judged by
        the address the resolver will actually connect to."""
        assert is_cloud_metadata_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["10.0.0.5", "backend.test", "localhost", "127.0.0.1", "::1", "192.168.1.20", "", None],
    )
    def test_ordinary_hosts_are_not_blocked(self, host: str | None) -> None:
        assert is_cloud_metadata_host(host) is False


class TestIsLoopbackHost:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.5", "::1", "[::1]"])
    def test_loopback(self, host: str) -> None:
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["10.0.0.5", "backend.test", "", None])
    def test_not_loopback(self, host: str | None) -> None:
        assert is_loopback_host(host) is False


class TestRedactUrlUserinfo:
    def test_strips_user_and_password(self) -> None:
        assert redact_url_userinfo(f"https://u:{VFY_MARKER}@h.test/x") == "https://h.test/x"

    def test_strips_username_only(self) -> None:
        assert redact_url_userinfo("https://u@h.test/x") == "https://h.test/x"

    def test_accepts_a_pydantic_url(self) -> None:
        # model_construct bypasses validate_url on purpose: this helper is
        # the defence in depth for exactly that bypass.
        config = BackendConfig.model_construct(
            type="jaeger", url=HttpUrl(f"https://u:{VFY_MARKER}@h.test/")
        )

        assert redact_url_userinfo(config.url) == "https://h.test/"

    def test_leaves_a_clean_url_alone(self) -> None:
        assert redact_url_userinfo("http://localhost:16686/") == "http://localhost:16686/"
        assert redact_url_userinfo("https://h.test/api?trace_id=abc") == (
            "https://h.test/api?trace_id=abc"
        )


class TestRedactUserinfoInText:
    def test_redacts_the_httpx_error_message_shape(self) -> None:
        text = (
            f"Client error '401 Unauthorized' for url 'https://u:{VFY_MARKER}@h.test/api/services'"
        )

        redacted = redact_userinfo_in_text(text)

        assert redacted is not None
        assert VFY_MARKER not in redacted
        assert "u:" not in redacted
        assert "https://[REDACTED]@h.test/api/services" in redacted
        assert redacted.startswith("Client error '401 Unauthorized'")

    def test_redacts_the_pydantic_input_value_echo(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            BackendConfig(type="jaeger", url=HttpUrl(f"https://u:{VFY_MARKER}@h.test/x"))
        raw = str(exc_info.value)
        assert VFY_MARKER in raw  # the premise: pydantic echoes input_value

        redacted = redact_userinfo_in_text(raw)

        assert redacted is not None
        assert VFY_MARKER not in redacted

    def test_redacts_every_url_in_the_text(self) -> None:
        text = f"primary https://a:{VFY_MARKER}@one.test/ secondary http://b:{VFY_MARKER}@two.test/"

        redacted = redact_userinfo_in_text(text)

        assert redacted == (
            "primary https://[REDACTED]@one.test/ secondary http://[REDACTED]@two.test/"
        )

    def test_username_only_userinfo_is_redacted_too(self) -> None:
        assert (
            redact_userinfo_in_text("see https://svc@h.test/") == "see https://[REDACTED]@h.test/"
        )

    def test_password_containing_at_is_redacted_up_to_the_last_at(self) -> None:
        """WHATWG (and so pydantic) accept a literal '@' inside the password:
        host h.test, password p%40<marker>. The userinfo therefore runs to
        the LAST '@' before the authority ends; stopping at the first one
        would leave the password's tail in place."""
        redacted = redact_userinfo_in_text(f"input_value='https://u:p@{VFY_MARKER}@h.test/x'")

        assert redacted == "input_value='https://[REDACTED]@h.test/x'"

    def test_pydantic_echo_of_a_password_containing_at_is_fully_redacted(self) -> None:
        """The string path is what from_env() takes, and the one message
        main()/doctor are guaranteed to emit for a rejected credential."""
        with pytest.raises(ValidationError) as exc_info:
            BackendConfig.model_validate(
                {"type": "jaeger", "url": f"https://u:p@{VFY_MARKER}@h.test/x"}
            )
        raw = str(exc_info.value)
        assert f"p@{VFY_MARKER}@h.test" in raw  # the premise: pydantic echoes input_value

        redacted = redact_userinfo_in_text(raw)

        assert redacted is not None
        assert VFY_MARKER not in redacted
        assert "p@" not in redacted
        assert "[REDACTED]@h.test/x" in redacted

    @pytest.mark.parametrize(
        "text",
        [
            "Client error '404 Not Found' for url 'https://h.test/api/traces/abc'",
            "docs live at https://example.com/docs@v2 and https://example.com/?user=a@b",
            "mail me at someone@example.com",
            "",
        ],
    )
    def test_text_without_userinfo_is_unchanged(self, text: str) -> None:
        assert redact_userinfo_in_text(text) == text

    def test_none_passes_through(self) -> None:
        assert redact_userinfo_in_text(None) is None


class TestUserinfoRedactingFilter:
    """The handler-level last line of defence: whatever logger produced the
    record, neither its message nor its formatted traceback may leave with a
    userinfo component."""

    def _record(self, msg: str, *args: object, exc_info: object = None) -> logging.LogRecord:
        return logging.LogRecord(
            name="third.party",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=args or None,
            exc_info=exc_info,  # type: ignore[arg-type]
        )

    def test_rewrites_a_message_with_args(self) -> None:
        record = self._record("HTTP Request: %s %s", "GET", f"https://u:{VFY_MARKER}@h.test/x")

        assert UserinfoRedactingFilter().filter(record) is True

        assert record.getMessage() == "HTTP Request: GET https://[REDACTED]@h.test/x"
        assert record.args is None
        assert logging.Formatter("%(message)s").format(record) == (
            "HTTP Request: GET https://[REDACTED]@h.test/x"
        )

    def test_redacts_a_real_validation_error_traceback(self) -> None:
        try:
            BackendConfig(type="jaeger", url=HttpUrl(f"https://u:{VFY_MARKER}@h.test/x"))
        except ValidationError:
            record = self._record("Server error: %s", "config failed", exc_info=sys.exc_info())

        UserinfoRedactingFilter().filter(record)

        assert record.exc_info is None  # Formatter must use the redacted cache, not re-render
        assert record.exc_text is not None
        assert "ValidationError" in record.exc_text
        assert VFY_MARKER not in record.exc_text
        formatted = logging.Formatter("%(message)s").format(record)
        assert VFY_MARKER not in formatted
        assert "Server error: config failed" in formatted

    def test_leaves_a_clean_record_untouched(self) -> None:
        record = self._record("Backend health check: %s", "healthy")

        UserinfoRedactingFilter().filter(record)

        assert record.msg == "Backend health check: %s"
        assert record.args == ("healthy",)
        assert record.exc_info is None
        assert record.exc_text is None

    def test_malformed_format_call_passes_without_raising_and_redacts_args(self) -> None:
        """Handler.handle() runs filters OUTSIDE emit()'s try/except, so an
        exception here would escape into the logging caller (any dependency
        with a mismatched %-format call) - where, without the filter,
        Handler.handleError would have swallowed it. The record is instead
        redacted piecewise so the Formatter failure that follows prints no
        userinfo either."""
        record = self._record("HTTP Request: %s %s", f"https://u:{VFY_MARKER}@h.test/x")

        assert UserinfoRedactingFilter().filter(record) is True

        assert record.args == ("https://[REDACTED]@h.test/x",)
        assert record.msg == "HTTP Request: %s %s"
        assert VFY_MARKER not in repr(record.__dict__)

    def test_malformed_mapping_args_are_redacted_per_value(self) -> None:
        record = self._record(
            "%(method)s %(url)s %(missing)s",
            {"method": "GET", "url": f"https://u:{VFY_MARKER}@h.test/x"},
        )

        assert UserinfoRedactingFilter().filter(record) is True

        assert isinstance(record.args, dict)
        assert record.args["url"] == "https://[REDACTED]@h.test/x"
        assert VFY_MARKER not in repr(record.__dict__)

    def test_malformed_format_call_reaches_handle_error_not_the_caller(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End to end through a real handler: the caller gets no exception,
        and the '--- Logging error ---' notice handleError prints to stderr
        (Message: %r / Arguments: %s) carries the redacted pieces only."""
        handler = logging.StreamHandler(io.StringIO())
        handler.addFilter(UserinfoRedactingFilter())
        log = logging.getLogger("tests.security.malformed_format")
        log.handlers = [handler]
        log.propagate = False
        try:
            log.error("%s %s", f"https://u:{VFY_MARKER}@h.test/x")  # one arg, two slots
        finally:
            log.handlers = []
            log.propagate = True

        err = capsys.readouterr().err
        assert "--- Logging error ---" in err
        assert "https://[REDACTED]@h.test/x" in err
        assert VFY_MARKER not in err

    def test_unrenderable_traceback_is_dropped_rather_than_raised(self) -> None:
        """A traceback the Formatter cannot render cannot be proven
        credential-free, and raising here would hit the caller: drop it."""
        record = self._record("Server error: %s", "x", exc_info=(None, None, "not a traceback"))

        assert UserinfoRedactingFilter().filter(record) is True

        assert record.exc_info is None
        assert record.exc_text == "<traceback unavailable>"
        assert logging.Formatter("%(message)s").format(record) == (
            "Server error: x\n<traceback unavailable>"
        )
