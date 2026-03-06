"""Tests for the Gmail API client module."""

import base64
import email
import email.mime.text
from unittest.mock import MagicMock

from email_triage.data.gmail import (
    get_message,
    get_message_headers,
    get_message_headers_batch,
    list_messages,
)


def _make_raw_email(subject: str = "Test Subject", body: str = "Hello world") -> str:
    """Create a base64url-encoded RFC 2822 email."""
    msg = email.mime.text.MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    raw_bytes = msg.as_bytes()
    return base64.urlsafe_b64encode(raw_bytes).decode("ascii")


def _mock_service():
    """Build a mock Gmail API service."""
    service = MagicMock()
    return service


class TestGetMessage:
    def test_parses_raw_email(self):
        service = _mock_service()
        raw = _make_raw_email(subject="Meeting Tomorrow", body="Please confirm.")
        service.users().messages().get().execute.return_value = {"raw": raw}

        result = get_message(service, "msg123")

        assert result["subject"] == "Meeting Tomorrow"
        assert "Please confirm." in result["body"]
        assert result["source_file"] == "gmail:msg123"

    def test_source_file_format(self):
        service = _mock_service()
        raw = _make_raw_email()
        service.users().messages().get().execute.return_value = {"raw": raw}

        result = get_message(service, "abc_XYZ-123")
        assert result["source_file"] == "gmail:abc_XYZ-123"

    def test_empty_subject(self):
        service = _mock_service()
        msg = email.mime.text.MIMEText("body text")
        msg["From"] = "sender@example.com"
        # No Subject header
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        service.users().messages().get().execute.return_value = {"raw": raw}

        result = get_message(service, "no_subj")
        assert result["subject"] == ""
        assert "body text" in result["body"]


class TestGetMessageHeaders:
    def test_extracts_headers(self):
        service = _mock_service()
        service.users().messages().get().execute.return_value = {
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test"},
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                ]
            },
            "snippet": "Preview text...",
        }

        result = get_message_headers(service, "hdr123")
        assert result["id"] == "hdr123"
        assert result["subject"] == "Test"
        assert result["from"] == "alice@example.com"
        assert result["date"] == "Mon, 1 Jan 2024 12:00:00 +0000"
        assert result["snippet"] == "Preview text..."

    def test_missing_headers(self):
        service = _mock_service()
        service.users().messages().get().execute.return_value = {
            "payload": {"headers": []},
            "snippet": "",
        }

        result = get_message_headers(service, "empty")
        assert result["subject"] == "(no subject)"
        assert result["from"] == ""


class TestGetMessageHeadersBatch:
    def test_fetches_multiple(self):
        service = _mock_service()
        service.users().messages().get().execute.side_effect = [
            {
                "payload": {"headers": [{"name": "Subject", "value": f"Email {i}"}]},
                "snippet": f"snippet {i}",
            }
            for i in range(3)
        ]

        results = get_message_headers_batch(service, ["a", "b", "c"])
        assert len(results) == 3
        assert results[0]["subject"] == "Email 0"
        assert results[2]["subject"] == "Email 2"


class TestListMessages:
    def test_basic_list(self):
        service = _mock_service()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "1"}, {"id": "2"}],
            "nextPageToken": "token_abc",
        }

        result = list_messages(service, query="from:boss", max_results=10)
        assert len(result["messages"]) == 2
        assert result["nextPageToken"] == "token_abc"

    def test_empty_results(self):
        service = _mock_service()
        service.users().messages().list().execute.return_value = {}

        result = list_messages(service)
        assert result["messages"] == []
        assert result["nextPageToken"] is None
