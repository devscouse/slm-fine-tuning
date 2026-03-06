"""Streamlit dashboard for email ingestion, labeling, and dataset management."""

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from email_triage.labels import CLASSES, CLASS_NAMES
from email_triage.data.email_parser import parse_directory, parse_email_bytes
from email_triage.data.gmail import (
    build_service,
    get_message,
    get_message_headers_batch,
    list_messages,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = PROJECT_ROOT / "secrets"
CREDENTIALS_PATH = SECRETS_DIR / "credentials.json"
TOKEN_PATH = SECRETS_DIR / "token.json"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
EMAILS_DIR = DATA_RAW / "emails"
EMAILS_PATH = DATA_RAW / "emails.jsonl"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic" / "emails.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Email Triage Labeling", layout="wide")

tab_upload, tab_gmail, tab_label, tab_manage, tab_stats = st.tabs(
    ["Upload", "Import from Gmail", "Label", "Browse & Manage", "Stats & Export"]
)

# ===================================================================
# PAGE 1: Upload
# ===================================================================
with tab_upload:
    st.header("Upload Email Files")

    uploaded_files = st.file_uploader(
        "Drop .eml or .msg files here",
        type=["eml", "msg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        parsed: list[dict] = []
        errors: list[str] = []

        for uf in uploaded_files:
            try:
                uf.seek(0)
                raw_bytes = uf.read()
                result = parse_email_bytes(raw_bytes, uf.name)
                result["_raw_bytes"] = raw_bytes
                parsed.append(result)
            except Exception as e:
                errors.append(f"{uf.name}: {e}")

        if errors:
            st.warning("Some files failed to parse:")
            for err in errors:
                st.text(f"  - {err}")

        if parsed:
            st.subheader(f"{len(parsed)} emails parsed")
            preview_data = [
                {
                    "Source File": r["source_file"],
                    "Subject": r["subject"],
                    "Body (snippet)": r["body"][:120]
                    + ("..." if len(r["body"]) > 120 else ""),
                }
                for r in parsed
            ]
            st.dataframe(preview_data, use_container_width=True)

            if st.button("Add to dataset"):
                existing_sources = {
                    rec.get("source_file", "") for rec in _read_jsonl(EMAILS_PATH)
                }

                new_records = [
                    r for r in parsed if r["source_file"] not in existing_sources
                ]
                dupes = len(parsed) - len(new_records)

                if new_records:
                    # Save raw email files to disk
                    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
                    for rec in new_records:
                        raw = rec.pop("_raw_bytes")
                        (EMAILS_DIR / rec["source_file"]).write_bytes(raw)
                    _append_jsonl(new_records, EMAILS_PATH)

                st.success(
                    f"{len(new_records)} new emails added, {dupes} duplicates skipped"
                )

# ===================================================================
# PAGE 2: Import from Gmail
# ===================================================================
with tab_gmail:
    st.header("Import from Gmail")

    if not CREDENTIALS_PATH.exists():
        st.warning("Gmail API credentials not found.")
        st.markdown(
            """
**One-time setup:**

1. Go to [Google Cloud Console — Credentials](https://console.cloud.google.com/apis/credentials)
2. Enable the **Gmail API** for your project
3. Create an **OAuth 2.0 Client ID** (Application type: *Desktop app*)
4. Download the JSON file
5. Save it as `secrets/credentials.json` in this project
"""
        )
    else:
        # --- Connect ---
        if st.button("Connect to Gmail"):
            try:
                st.session_state.gmail_service = build_service(
                    CREDENTIALS_PATH, TOKEN_PATH
                )
                st.success("Connected to Gmail!")
            except Exception as e:
                st.error(f"Authentication failed: {e}")

        if "gmail_service" in st.session_state:
            service = st.session_state.gmail_service

            # --- Search ---
            gmail_query = st.text_input(
                "Search Gmail",
                placeholder="e.g. from:boss@example.com after:2024/01/01",
                help="Uses Gmail search syntax",
            )

            if "gmail_page_tokens" not in st.session_state:
                st.session_state.gmail_page_tokens = [None]

            if st.button("Search") or gmail_query:
                # Reset pagination on new search
                if (
                    "gmail_last_query" not in st.session_state
                    or st.session_state.gmail_last_query != gmail_query
                ):
                    st.session_state.gmail_page_tokens = [None]
                    st.session_state.gmail_last_query = gmail_query

                current_token = st.session_state.gmail_page_tokens[-1]
                try:
                    result = list_messages(
                        service,
                        query=gmail_query,
                        max_results=25,
                        page_token=current_token,
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    result = None

                if result and result["messages"]:
                    msg_ids = [m["id"] for m in result["messages"]]
                    try:
                        headers = get_message_headers_batch(service, msg_ids)
                    except Exception as e:
                        st.error(f"Failed to fetch message headers: {e}")
                        headers = []

                    if headers:
                        # --- Message list with checkboxes ---
                        st.write(f"Showing {len(headers)} messages")

                        if "gmail_selected" not in st.session_state:
                            st.session_state.gmail_selected = set()

                        for hdr in headers:
                            col_check, col_info = st.columns([1, 20])
                            with col_check:
                                selected = st.checkbox(
                                    "sel",
                                    key=f"gmail_sel_{hdr['id']}",
                                    label_visibility="collapsed",
                                )
                                if selected:
                                    st.session_state.gmail_selected.add(hdr["id"])
                                else:
                                    st.session_state.gmail_selected.discard(hdr["id"])
                            with col_info:
                                st.markdown(
                                    f"**{hdr['subject']}**  \n"
                                    f"From: {hdr['from']} | {hdr['date']}  \n"
                                    f"_{hdr['snippet']}_"
                                )

                        # --- Pagination ---
                        nav_cols = st.columns(3)
                        with nav_cols[0]:
                            if len(st.session_state.gmail_page_tokens) > 1:
                                if st.button("Previous"):
                                    st.session_state.gmail_page_tokens.pop()
                                    st.rerun()
                        with nav_cols[2]:
                            next_token = result.get("nextPageToken")
                            if next_token:
                                if st.button("Next"):
                                    st.session_state.gmail_page_tokens.append(
                                        next_token
                                    )
                                    st.rerun()

                        # --- Import button ---
                        selected_ids = st.session_state.gmail_selected
                        if selected_ids:
                            if st.button(
                                f"Import {len(selected_ids)} selected email(s)"
                            ):
                                existing_sources = {
                                    rec.get("source_file", "")
                                    for rec in _read_jsonl(EMAILS_PATH)
                                }

                                imported = 0
                                skipped = 0
                                errors = []
                                for mid in selected_ids:
                                    source_key = f"gmail:{mid}"
                                    if source_key in existing_sources:
                                        skipped += 1
                                        continue
                                    try:
                                        record = get_message(service, mid)
                                        _append_jsonl([record], EMAILS_PATH)
                                        imported += 1
                                    except Exception as e:
                                        errors.append(f"{mid}: {e}")

                                st.success(
                                    f"Imported {imported} email(s), "
                                    f"{skipped} duplicate(s) skipped."
                                )
                                if errors:
                                    st.warning("Some imports failed:")
                                    for err in errors:
                                        st.text(f"  - {err}")

                                st.session_state.gmail_selected = set()
                elif result:
                    st.info("No messages found for that search.")

# ===================================================================
# PAGE 3: Label
# ===================================================================
with tab_label:
    st.header("Label Emails")

    # Load all records, filter to those without a label
    all_emails = _read_jsonl(EMAILS_PATH)
    unlabeled = [r for r in all_emails if "label" not in r]

    if "session_labeled" not in st.session_state:
        st.session_state.session_labeled = 0

    if not unlabeled:
        if st.session_state.session_labeled > 0:
            st.success(
                f"All done! Labeled {st.session_state.session_labeled} emails this session."
            )
        else:
            st.info("No unlabeled emails. Upload some files first.")
    else:
        remaining = len(unlabeled)
        labeled = st.session_state.session_labeled
        st.progress(
            labeled / (labeled + remaining),
            text=f"{remaining} remaining ({labeled} labeled this session)",
        )

        # Always show the front of the queue
        record = unlabeled[0]

        st.subheader(record.get("subject", "(no subject)"))
        with st.container(height=500, border=True):
            st.markdown(record.get("body", ""))

        st.write("**Assign a class:**")
        cols = st.columns(len(CLASSES) + 1)

        for i, cls in enumerate(CLASSES):
            if cols[i].button(
                f"{cls.name}",
                help=cls.description,
                key=f"label_btn_{cls.name}",
                use_container_width=True,
            ):
                record["label"] = cls.name
                _write_jsonl(all_emails, EMAILS_PATH)
                st.session_state.session_labeled += 1
                st.rerun()

        if cols[-1].button("Skip", key="label_skip", use_container_width=True):
            # Rotate: move record to end of the full list so next
            # unlabeled email appears at the front of the queue.
            all_emails.remove(record)
            all_emails.append(record)
            _write_jsonl(all_emails, EMAILS_PATH)
            st.rerun()

            with st.expander("Class definitions & examples"):
                _class_examples = {
                    "attention": [
                        "An email expecting or requesting a reply",
                        "A request to make a payment (invoice, bill, subscription renewal)",
                        "A calendar invite or scheduling request that needs an RSVP",
                        "An approval request (expense report, PR review, access request)",
                        "A deadline or due-date reminder",
                    ],
                    "notice": [
                        "Newsletters or curated content digests",
                        "A reply that does not call for further action",
                        "Account statements or billing summaries",
                        "FYI or informational forwards from colleagues",
                    ],
                    "ignore": [
                        "Order confirmations and purchase receipts",
                        "Shipping or delivery status updates",
                        "Marketing offers and promotional emails",
                        "Social media notifications",
                        "Automated system notifications (CI/CD, cron jobs)",
                    ],
                    "security": [
                        "Multi-factor authentication (MFA) codes",
                        "Email address verification requests",
                        "Security alerts (unusual login, account locked)",
                        "Password reset or change requests",
                        "Login from a new device notifications",
                    ],
                }
                for cls in CLASSES:
                    st.markdown(f"**{cls.name.capitalize()}** — {cls.description}")
                    examples = _class_examples.get(cls.name, [])
                    for ex in examples:
                        st.markdown(f"- {ex}")

# ===================================================================
# PAGE 4: Browse & Manage
# ===================================================================
with tab_manage:
    st.header("Browse & Manage Emails")

    all_records = _read_jsonl(EMAILS_PATH)

    if not all_records:
        st.info("No emails yet. Upload some files first.")
    else:
        # --- Re-parse button ---
        if st.button("Re-parse all files"):
            if EMAILS_DIR.exists():
                parsed = parse_directory(EMAILS_DIR)
                existing_by_src = {r.get("source_file", ""): r for r in all_records}
                merged: list[dict] = []
                for rec in parsed:
                    src = rec.get("source_file", "")
                    if src in existing_by_src:
                        # Update subject/body but keep existing label + other fields
                        existing = existing_by_src.pop(src)
                        existing["subject"] = rec["subject"]
                        existing["body"] = rec["body"]
                        merged.append(existing)
                    else:
                        merged.append(rec)
                # Keep any records not matched by source_file (e.g. manually added)
                merged.extend(existing_by_src.values())
                _write_jsonl(merged, EMAILS_PATH)
                st.success(f"Re-parsed: {len(merged)} records updated")
                st.rerun()
            else:
                st.warning(f"No emails directory found at {EMAILS_DIR}")

        # --- Filters ---
        filter_options = CLASS_NAMES + ["unlabeled"]
        col_filter, col_search = st.columns(2)
        with col_filter:
            selected_classes = st.multiselect(
                "Filter by class", filter_options, default=filter_options
            )
        with col_search:
            search_text = st.text_input("Search subject/body")

        filtered = [
            r
            for r in all_records
            if (r.get("label", "unlabeled") in selected_classes)
            and (
                not search_text
                or search_text.lower() in r.get("subject", "").lower()
                or search_text.lower() in r.get("body", "").lower()
            )
        ]

        st.write(f"Showing {len(filtered)} of {len(all_records)} records")

        for i, record in enumerate(filtered):
            label_tag = record.get("label", "").upper() or "UNLABELED"
            subject = record.get("subject", "(no subject)")

            # Row: expander + delete button side-by-side
            col_expand, col_del = st.columns([20, 1])

            with col_del:
                if st.button("\u2716", key=f"delete_{i}", help="Delete this email"):
                    all_records = [
                        rec
                        for rec in all_records
                        if not (
                            rec.get("source_file") == record.get("source_file")
                            and rec.get("subject") == record.get("subject")
                        )
                    ]
                    _write_jsonl(all_records, EMAILS_PATH)
                    st.success("Record deleted")
                    st.rerun()

            with col_expand:
                with st.expander(f"[{label_tag}] {subject}"):
                    with st.container(height=400, border=True):
                        st.markdown(record.get("body", ""))

                    current_label = record.get("label", "")
                    label_options = ["unlabeled"] + CLASS_NAMES
                    current_idx = (
                        label_options.index(current_label)
                        if current_label in label_options
                        else 0  # default to "unlabeled"
                    )

                    new_label = st.selectbox(
                        "Label",
                        label_options,
                        index=current_idx,
                        key=f"relabel_{i}",
                    )

                    # Auto-save when label changes
                    if new_label != current_label and (
                        new_label != "unlabeled" or current_label != ""
                    ):
                        for rec in all_records:
                            if rec.get("source_file") == record.get(
                                "source_file"
                            ) and rec.get("subject") == record.get("subject"):
                                if new_label == "unlabeled":
                                    rec.pop("label", None)
                                else:
                                    rec["label"] = new_label
                                break
                        _write_jsonl(all_records, EMAILS_PATH)
                        st.success(f"Updated to '{new_label}'")
                        st.rerun()

# ===================================================================
# PAGE 5: Stats & Export
# ===================================================================
with tab_stats:
    st.header("Dataset Statistics & Export")

    all_emails = _read_jsonl(EMAILS_PATH)
    labeled = [r for r in all_emails if "label" in r]
    synthetic = _read_jsonl(SYNTHETIC_PATH)

    # --- Labeled (real) data stats ---
    st.subheader("Real (Labeled) Data")
    if not labeled:
        st.info("No labeled emails yet.")
    else:
        label_counts = {}
        for r in labeled:
            lbl = r.get("label", "unknown")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        total = len(labeled)
        st.metric("Total", total)

        chart_cols = st.columns(len(CLASS_NAMES))
        for i, cls in enumerate(CLASS_NAMES):
            count = label_counts.get(cls, 0)
            pct = f"{count / total * 100:.1f}%" if total else "0%"
            chart_cols[i].metric(cls.capitalize(), f"{count} ({pct})")

        st.bar_chart(
            {cls: label_counts.get(cls, 0) for cls in CLASS_NAMES},
        )

    # --- Synthetic data stats ---
    if synthetic:
        st.subheader("Synthetic Data")
        synth_counts = {}
        for r in synthetic:
            lbl = r.get("label", "unknown")
            synth_counts[lbl] = synth_counts.get(lbl, 0) + 1

        st.metric("Total synthetic", len(synthetic))
        synth_cols = st.columns(len(CLASS_NAMES))
        for i, cls in enumerate(CLASS_NAMES):
            count = synth_counts.get(cls, 0)
            synth_cols[i].metric(cls.capitalize(), count)

    # --- Combined stats ---
    combined = labeled + synthetic
    if combined:
        st.subheader("Combined (Real + Synthetic)")
        comb_counts = {}
        for r in combined:
            lbl = r.get("label", "unknown")
            comb_counts[lbl] = comb_counts.get(lbl, 0) + 1
        st.metric("Total combined", len(combined))
        comb_cols = st.columns(len(CLASS_NAMES))
        for i, cls in enumerate(CLASS_NAMES):
            count = comb_counts.get(cls, 0)
            comb_cols[i].metric(cls.capitalize(), count)

    # --- Export / Split ---
    st.subheader("Export / Split")

    input_files: list[str] = []
    if labeled:
        input_files.append(str(EMAILS_PATH))
    if synthetic:
        input_files.append(str(SYNTHETIC_PATH))

    if not input_files:
        st.warning(
            "No data to split. Label some emails or generate synthetic data first."
        )
    else:
        st.write(f"Sources: {', '.join(Path(f).name for f in input_files)}")

        if st.button("Run Train/Val/Test Split"):
            from sklearn.model_selection import train_test_split
            import hashlib

            all_records = []
            for fp in input_files:
                all_records.extend(_read_jsonl(Path(fp)))

            # Only include labeled records for splitting
            all_records = [r for r in all_records if "label" in r]

            # Deduplicate by (subject, body) hash
            seen: set[str] = set()
            deduped: list[dict] = []
            for r in all_records:
                key = hashlib.sha256(
                    (r.get("subject", "") + r.get("body", "")).encode()
                ).hexdigest()
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)

            if len(deduped) < 10:
                st.error("Need at least 10 records for a meaningful split.")
            else:
                labels = [r["label"] for r in deduped]
                train, temp, train_labels, temp_labels = train_test_split(
                    deduped,
                    labels,
                    test_size=0.30,
                    random_state=42,
                    stratify=labels,
                )
                val, test = train_test_split(
                    temp,
                    test_size=0.50,
                    random_state=42,
                    stratify=temp_labels,
                )

                _write_jsonl(train, PROCESSED_DIR / "train.jsonl")
                _write_jsonl(val, PROCESSED_DIR / "val.jsonl")
                _write_jsonl(test, PROCESSED_DIR / "test.jsonl")

                st.success("Split complete!")
                res_cols = st.columns(4)
                res_cols[0].metric("Total (deduped)", len(deduped))
                res_cols[1].metric("Train", len(train))
                res_cols[2].metric("Val", len(val))
                res_cols[3].metric("Test", len(test))
