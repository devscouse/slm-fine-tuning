"""Tests for email_triage.data.email_parser."""

from unittest.mock import MagicMock, patch

import pytest

from email_triage.data.email_parser import (
    _clean_text,
    parse_directory,
    parse_email_bytes,
    parse_email_file,
)


def _make_eml(subject: str, body: str, content_type: str = "text/plain") -> bytes:
    """Build a minimal single-part .eml as raw bytes."""
    return (
        f"From: sender@example.com\r\n"
        f"To: recipient@example.com\r\n"
        f"Subject: {subject}\r\n"
        f"Content-Type: {content_type}; charset=utf-8\r\n"
        f"\r\n"
        f"{body}"
    ).encode("utf-8")


def _make_multipart_eml(subject: str, plain: str, html: str) -> bytes:
    """Build a multipart/alternative .eml with both text/plain and text/html."""
    boundary = "----=_Part_boundary"
    return (
        f"From: sender@example.com\r\n"
        f"To: recipient@example.com\r\n"
        f"Subject: {subject}\r\n"
        f'Content-Type: multipart/alternative; boundary="{boundary}"\r\n'
        f"\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{plain}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: text/html; charset=utf-8\r\n"
        f"\r\n"
        f"{html}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")


class TestParseEmlPlainText:
    def test_basic(self):
        data = _make_eml("Test Subject", "Hello, world!")
        result = parse_email_bytes(data, "test.eml")
        assert result["subject"] == "Test Subject"
        assert result["body"] == "Hello, world!"
        assert result["source_file"] == "test.eml"


class TestParseEmlMultipart:
    def test_prefers_plain_text(self):
        data = _make_multipart_eml(
            "Multi", "Plain text body", "<p>HTML body</p>"
        )
        result = parse_email_bytes(data, "multi.eml")
        assert result["subject"] == "Multi"
        assert result["body"] == "Plain text body"


class TestParseEmlHtmlOnly:
    def test_extracts_plain_text(self):
        html_body = "<html><body><p>Hello</p><br><b>World</b></body></html>"
        data = _make_eml("HTML Only", html_body, content_type="text/html")
        result = parse_email_bytes(data, "html.eml")
        assert result["subject"] == "HTML Only"
        assert "Hello" in result["body"]
        assert "World" in result["body"]
        assert "<p>" not in result["body"]

    def test_strips_style_blocks(self):
        html_body = "<html><head><style>body{color:red}</style></head><body><p>Content</p></body></html>"
        data = _make_eml("Styled", html_body, content_type="text/html")
        result = parse_email_bytes(data, "styled.eml")
        assert "color:red" not in result["body"]
        assert "Content" in result["body"]


class TestParseMsg:
    @patch("email_triage.data.email_parser.extract_msg")
    def test_basic(self, mock_extract_msg):
        mock_msg = MagicMock()
        mock_msg.subject = "MSG Subject"
        mock_msg.body = "MSG body text"
        mock_extract_msg.openMsg.return_value = mock_msg

        result = parse_email_bytes(b"fake msg data", "test.msg")
        assert result["subject"] == "MSG Subject"
        assert result["body"] == "MSG body text"
        assert result["source_file"] == "test.msg"
        mock_msg.close.assert_called_once()


class TestParseMsgHtmlFallback:
    @patch("email_triage.data.email_parser.extract_msg")
    def test_html_body_used_when_plain_is_none(self, mock_extract_msg):
        mock_msg = MagicMock()
        mock_msg.subject = "HTML Only"
        mock_msg.body = None
        mock_msg.htmlBody = b"<h1>Hello</h1><p>World</p>"
        mock_extract_msg.openMsg.return_value = mock_msg

        result = parse_email_bytes(b"fake msg data", "html.msg")
        assert result["subject"] == "HTML Only"
        assert "Hello" in result["body"]
        assert "World" in result["body"]
        assert "<p>" not in result["body"]
        mock_msg.close.assert_called_once()

    @patch("email_triage.data.email_parser.extract_msg")
    def test_plain_body_preferred_over_html(self, mock_extract_msg):
        mock_msg = MagicMock()
        mock_msg.subject = "Both"
        mock_msg.body = "Plain text"
        mock_msg.htmlBody = b"<p>HTML text</p>"
        mock_extract_msg.openMsg.return_value = mock_msg

        result = parse_email_bytes(b"fake msg data", "both.msg")
        assert result["body"] == "Plain text"
        mock_msg.close.assert_called_once()


class TestParseMsgNullBytes:
    @patch("email_triage.data.email_parser.extract_msg")
    def test_null_bytes_stripped_from_subject(self, mock_extract_msg):
        mock_msg = MagicMock()
        mock_msg.subject = "Hello\x00World\x00"
        mock_msg.body = "body"
        mock_extract_msg.openMsg.return_value = mock_msg

        result = parse_email_bytes(b"fake msg data", "null.msg")
        assert result["subject"] == "HelloWorld"
        mock_msg.close.assert_called_once()


class TestUnsupportedExtension:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_email_bytes(b"data", "file.txt")


class TestParseEmailBytes:
    def test_eml_in_memory(self):
        data = _make_eml("Bytes Test", "Body content")
        result = parse_email_bytes(data, "memory.eml")
        assert result["subject"] == "Bytes Test"
        assert result["body"] == "Body content"
        assert result["source_file"] == "memory.eml"


class TestParseDirectory:
    def test_parses_all_eml_files(self, tmp_path):
        (tmp_path / "a.eml").write_bytes(_make_eml("Email A", "Body A"))
        (tmp_path / "b.eml").write_bytes(_make_eml("Email B", "Body B"))

        results = parse_directory(tmp_path)
        assert len(results) == 2
        subjects = {r["subject"] for r in results}
        assert subjects == {"Email A", "Email B"}


class TestEmptySubjectOrBody:
    def test_empty_subject(self):
        data = _make_eml("", "Some body")
        result = parse_email_bytes(data, "no_subject.eml")
        assert result["subject"] == ""
        assert result["body"] == "Some body"

    def test_empty_body(self):
        data = _make_eml("Has Subject", "")
        result = parse_email_bytes(data, "no_body.eml")
        assert result["subject"] == "Has Subject"
        assert result["body"] == ""


class TestCleanText:
    def test_strips_zero_width_chars(self):
        text = "hello\u200b\u200c\u200dworld"
        assert _clean_text(text) == "helloworld"

    def test_strips_soft_hyphens_and_bom(self):
        text = "soft\u00adhy\u00adphen and \ufeffBOM"
        assert _clean_text(text) == "softhyphen and BOM"

    def test_strips_directional_marks(self):
        text = "left\u200eright\u200ftext"
        assert _clean_text(text) == "leftrighttext"

    def test_collapses_hrule_dashes(self):
        text = "above\n----------\nbelow"
        assert _clean_text(text) == "above\n---\nbelow"

    def test_collapses_hrule_underscores(self):
        text = "above\n___________\nbelow"
        assert _clean_text(text) == "above\n---\nbelow"

    def test_collapses_hrule_equals(self):
        text = "above\n==========\nbelow"
        assert _clean_text(text) == "above\n---\nbelow"

    def test_collapses_inline_whitespace(self):
        text = "hello     world\tfoo\t\tbar"
        assert _clean_text(text) == "hello world foo bar"

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\n\nb"
        assert _clean_text(text) == "a\n\nb"

    def test_strips_leading_trailing_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_combined_cleanup(self):
        text = "  \u200bhello\u200c   world\n--------\n\n\n\nbye\ufeff  "
        result = _clean_text(text)
        assert result == "hello world\n---\n\nbye"


class TestCleanTextEndToEnd:
    def test_html_email_with_invisible_chars(self):
        html = (
            "<html><body>"
            "<p>Hello\u200b \u200cWorld</p>"
            "<hr>"
            "<p>Content\u00ad here</p>"
            "</body></html>"
        )
        data = _make_eml("Test", html, content_type="text/html")
        result = parse_email_bytes(data, "dirty.eml")
        assert "\u200b" not in result["body"]
        assert "\u200c" not in result["body"]
        assert "\u00ad" not in result["body"]
        assert "Hello" in result["body"]
        assert "Content" in result["body"]


class TestParseEmailFile:
    def test_from_disk(self, tmp_path):
        eml_path = tmp_path / "disk.eml"
        eml_path.write_bytes(_make_eml("Disk Test", "Disk body"))

        result = parse_email_file(eml_path)
        assert result["subject"] == "Disk Test"
        assert result["body"] == "Disk body"
        assert result["source_file"] == "disk.eml"
