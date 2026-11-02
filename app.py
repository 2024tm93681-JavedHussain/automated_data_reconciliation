import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Automated Data Reconciliation", page_icon="📊", layout="wide")

FIELDS = [
    "Sl No", "Tran Date", "DESC", "Travel Date", "PNR", "Name", "Age", "Gender",
    "Nationality", "From", "To", "Ticket No", "Class", "Booked By", "Payment Mode",
    "Order ID", "Party Name", "Location", "Fare", "Discount", "PSF",
    "YQ(Fuel Surcharge)", "Canc Charge", "ADC", "Net Sale", "Remarks", "Note"
]
KEY_FIELDS = [
    "Booked By", "Payment Mode", "Party Name", "Location", "Fare", "Discount",
    "PSF", "YQ(Fuel Surcharge)", "Canc Charge", "ADC", "Net Sale"
]
DATE_FIELDS = ["Tran Date", "Travel Date"]
NUMERIC_FIELDS = ["Age", "Fare", "Discount", "PSF", "YQ(Fuel Surcharge)", "Canc Charge", "ADC", "Net Sale"]

DEFAULT_RULES = {
    "all_fields_required": True,
    "online_order_id": True,
    "ticket_continuity": True,
    "desc_classification": True,
    "travel_net_sale": True,
    "cancel_net_sale": True,
    "duplicate_ticket": True,
    "duplicate_pnr": True,
    "duplicate_order_id": True,
    "duplicate_transaction": True,
    "duplicate_passenger_transaction": True,
    "net_sale_gt_fare": True,
}

DESC_TYPES = ["TRAVEL", "CANCEL", "INVALID", "RESCHEDULE", "DOWNGRADE", "UPGRADE", "RES + UG", "RES + DG", "VOID"]


def blank_mask(values):
    """Return a boolean mask for blank/null values in a Series or DataFrame."""
    null_tokens = ["nan", "none", "null", "na"]
    if isinstance(values, pd.DataFrame):
        normalized = values.astype("string").apply(lambda col: col.str.strip().str.lower())
        return values.isna() | normalized.eq("") | normalized.isin(null_tokens)

    normalized = values.astype("string").str.strip().str.lower()
    return values.isna() | normalized.eq("") | normalized.isin(null_tokens)


def standardize(df):
    df = df.copy()
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename = {lookup[f.lower()]: f for f in FIELDS if f.lower() in lookup}
    df = df.rename(columns=rename)
    for f in FIELDS:
        if f not in df.columns:
            df[f] = pd.NA
    return df[FIELDS]


def parse_dates(df):
    df = df.copy()
    for c in DATE_FIELDS:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    for c in NUMERIC_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ticket_parts(value):
    s = "" if pd.isna(value) else str(value).strip()
    m = re.match(r"^(.*?)(\d+)$", s)
    return (m.group(1), int(m.group(2))) if m else (s, None)


def add_issue(df, mask, code, detail):
    if not mask.any():
        return
    df.loc[mask, "Issue Count"] += 1
    df.loc[mask, "Issue Codes"] = df.loc[mask, "Issue Codes"].apply(lambda x: (x + "; " if x else "") + code)
    df.loc[mask, "Issue Details"] = df.loc[mask, "Issue Details"].apply(lambda x: (x + " | " if x else "") + detail)


def ticket_series_analysis(df, sequence_start=None, sequence_prefix=None):
    """Check duplicate tickets and plausible ticket-series gaps.

    Ticket numbers with the same visible prefix can belong to different series
    (for example, A412-12345 and A412-89981234). Keep those series separate.
    Very large jumps are reported as possible series changes, not counted as
    millions of missing tickets without evidence that the series is continuous.
    """
    ticket_text = df["Ticket No"].astype("string").fillna("").str.strip()
    parts = ticket_text.str.extract(r"^(.*?)(\d+)$")
    work = pd.DataFrame(index=df.index)
    work["TranDay"] = pd.to_datetime(df["Tran Date"], errors="coerce").dt.date
    work["TicketText"] = ticket_text
    work["Prefix"] = parts[0].fillna(ticket_text)
    work["Seq"] = pd.to_numeric(parts[1], errors="coerce")
    work["SuffixDigits"] = parts[1].str.len()

    valid_ticket = ticket_text.ne("") & work["Seq"].notna() & work["TranDay"].notna()
    duplicate_mask = ticket_text.isin(ticket_text[valid_ticket & ticket_text.duplicated(keep=False)])
    sequence_mask = pd.Series(False, index=df.index)
    missing_records, sequence_records = [], []

    # A large gap may mean a different ticket-number series or test/manual data.
    # Keep it in the review list but do not count the entire range as missing.
    MAX_COUNTABLE_GAP = 100

    if not valid_ticket.any():
        return duplicate_mask, sequence_mask, pd.DataFrame(columns=[
            "Tran Date", "Prefix", "Missing Count", "Missing Ticket Numbers", "Gap Type"
        ]), pd.DataFrame(columns=["rowidx", "Tran Date", "Ticket No", "Expected", "Issue Type", "Gap Size"])

    valid = work.loc[valid_ticket].copy()
    valid["Seq"] = valid["Seq"].astype("int64")
    valid["SuffixDigits"] = valid["SuffixDigits"].astype("int64")
    series_cols = ["TranDay", "Prefix", "SuffixDigits"]
    grouped = valid.sort_values(series_cols + ["Seq"]).groupby(series_cols, sort=False, dropna=False)

    configured_prefix = str(sequence_prefix or "").strip().upper().rstrip("-")
    for (day, prefix, suffix_digits), group in grouped:
        seqs = group["Seq"].to_numpy()
        if len(seqs) == 0:
            continue

        # Use a configured start only for the matching prefix and numeric width.
        prefix_matches = configured_prefix and str(prefix).strip().upper().rstrip("-") == configured_prefix
        if sequence_start is not None and prefix_matches and len(str(int(sequence_start))) == int(suffix_digits):
            expected_start = int(sequence_start)
        else:
            expected_start = int(seqs[0])

        first = int(seqs[0])
        if first > expected_start:
            gap = first - expected_start
            idx = group.index[0]
            if gap <= MAX_COUNTABLE_GAP:
                missing_records.append({
                    "Tran Date": day, "Prefix": prefix, "Missing Count": gap,
                    "Missing Ticket Numbers": ", ".join(
                        f"{prefix}{n}" for n in range(expected_start, first)
                    ), "Gap Type": "Expected start gap"
                })
            else:
                sequence_records.append({
                    "rowidx": idx, "Tran Date": day, "Ticket No": work.loc[idx, "TicketText"],
                    "Expected": f"{prefix}{expected_start}", "Issue Type": "Large start gap — verify series",
                    "Gap Size": gap
                })
            sequence_mask.loc[idx] = True

        for pos in range(1, len(seqs)):
            prev, current = int(seqs[pos - 1]), int(seqs[pos])
            # Duplicate ticket numbers are checked by the duplicate rule; do not
            # also report them as sequence breaks.
            if current == prev:
                continue
            if current == prev + 1:
                continue

            idx = group.index[pos]
            gap = current - prev - 1
            sequence_mask.loc[idx] = True
            if gap > 0 and gap <= MAX_COUNTABLE_GAP:
                missing_records.append({
                    "Tran Date": day, "Prefix": prefix, "Missing Count": gap,
                    "Missing Ticket Numbers": ", ".join(
                        f"{prefix}{n}" for n in range(prev + 1, current)
                    ), "Gap Type": "Sequence gap"
                })
                issue_type = "Missing ticket number(s)"
            elif gap > MAX_COUNTABLE_GAP:
                issue_type = "Large jump — verify series"
            else:
                issue_type = "Duplicate or non-increasing sequence"

            sequence_records.append({
                "rowidx": idx, "Tran Date": day, "Ticket No": work.loc[idx, "TicketText"],
                "Expected": f"{prefix}{prev + 1}", "Issue Type": issue_type, "Gap Size": max(gap, 0)
            })

    missing_columns = ["Tran Date", "Prefix", "Missing Count", "Missing Ticket Numbers", "Gap Type"]
    sequence_columns = ["rowidx", "Tran Date", "Ticket No", "Expected", "Issue Type", "Gap Size"]
    return (
        duplicate_mask,
        sequence_mask,
        pd.DataFrame(missing_records, columns=missing_columns),
        pd.DataFrame(sequence_records, columns=sequence_columns),
    )


def _canonical_value(value, column):
    """Normalise a value for duplicate-key comparison without altering source data."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if column in DATE_FIELDS:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
        return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
    if column in NUMERIC_FIELDS:
        number = pd.to_numeric(value, errors="coerce")
        return "" if pd.isna(number) else f"{float(number):.2f}"
    text = str(value).strip().upper()
    return "" if text in {"NAN", "NONE", "NULL", "NA", "<NA>"} else text


def duplicate_identifier_mask(series):
    """Mark every nonblank occurrence of a repeated identifier (case/space-insensitive)."""
    normalized = series.astype("string").str.strip().str.upper()
    valid = normalized.notna() & ~normalized.isin(["", "NAN", "NONE", "NULL", "NA", "<NA>"])
    result = pd.Series(False, index=series.index, dtype=bool)
    result.loc[valid] = normalized.loc[valid].duplicated(keep=False)
    return result


def composite_duplicate_mask(df, columns, required_columns=None):
    """Mark rows with a repeated composite business key; blank identifiers are not grouped."""
    normalized = pd.DataFrame(
        {column: df[column].map(lambda value: _canonical_value(value, column)) for column in columns},
        index=df.index,
    )
    required = required_columns or columns
    valid = normalized[required].ne("").all(axis=1)
    result = pd.Series(False, index=df.index, dtype=bool)
    if valid.any():
        result.loc[valid] = normalized.loc[valid].duplicated(keep=False)
    return result


def validate_all(raw, rules=None, sequence_start=None, sequence_prefix=None, custom_rules=None):
    """Run the six fixed reconciliation rules requested by the Accounts Team."""
    df = parse_dates(standardize(raw))
    df["Source Row"] = range(2, len(df) + 2)
    df["Issue Count"] = 0
    df["Issue Codes"] = ""
    df["Issue Details"] = ""
    df["Expected Net Sale"] = pd.NA
    df["Calculated Gap"] = pd.NA
    df["Ticket Sequence Check"] = "Not Checked"
    df["Ticket Duplicate Check"] = "Not Checked"
    df["PNR Duplicate Check"] = "Not Checked"
    df["Order ID Duplicate Check"] = "Not Checked"
    df["Transaction Duplicate Check"] = "Not Checked"
    df["Passenger Transaction Duplicate Check"] = "Not Checked"
    df["Net Sale > Fare Check"] = "Not Checked"
    rules = rules or {}

    # Compute ticket-series checks before using the sequence mask below.
    _, sequence_break_mask, missing_ticket_df, sequence_df = ticket_series_analysis(df)

    # Duplicate ticket validation applies to TRAVEL rows only.
    # Shared PNRs are valid, and CANCEL rows may reuse the original ticket number.
    desc_upper = df["DESC"].astype("string").str.strip().str.upper()
    travel = desc_upper.eq("TRAVEL")
    travel_ticket = df["Ticket No"].astype("string").str.strip().str.upper()
    valid_travel_ticket = travel & ~blank_mask(df["Ticket No"])
    duplicate_ticket_mask = pd.Series(False, index=df.index, dtype=bool)
    duplicate_ticket_mask.loc[valid_travel_ticket] = travel_ticket.loc[valid_travel_ticket].duplicated(keep=False)
    df.loc[~blank_mask(df["PNR"]), "PNR Duplicate Check"] = "Shared PNR Allowed"
    add_issue(df, duplicate_ticket_mask, "DUPLICATE_TICKET",
              "Ticket No is repeated among TRAVEL transactions; verify ticket uniqueness")
    df.loc[duplicate_ticket_mask, "Ticket Duplicate Check"] = "Duplicate"
    df.loc[valid_travel_ticket & ~duplicate_ticket_mask, "Ticket Duplicate Check"] = "Unique"
    # Do not treat repeated Order IDs, shared PNRs, or CANCEL/TRAVEL ticket reuse as duplicate transactions.
    df["Order ID Duplicate Check"] = "Not Used"
    df["Transaction Duplicate Check"] = "Not Used"
    df["Passenger Transaction Duplicate Check"] = "Not Used"

    # Required-field validation checks all report fields except Order ID.
    for column in [field for field in FIELDS if field != "Order ID"]:
        add_issue(df, blank_mask(df[column]), "BLANK_FIELD",
                  f"{column} is blank/null/empty")

    # Blank Order ID is a dedicated exception only when all three online fields equal Online.
    online_mask = (
        df["Payment Mode"].astype("string").str.strip().str.casefold().eq("online").fillna(False)
        & df["Party Name"].astype("string").str.strip().str.casefold().eq("online").fillna(False)
        & df["Location"].astype("string").str.strip().str.casefold().eq("online").fillna(False)
    )
    blank_online_order_id = online_mask & blank_mask(df["Order ID"])
    add_issue(df, blank_online_order_id, "ONLINE_ORDER_ID_MISSING",
              "Blank Order ID for Online Payment Mode, Party Name and Location")

    # Rule 4: ticket-series continuity, grouped by transaction day, prefix and
    # numeric suffix width. No manually configured first number or prefix is used.
    add_issue(df, sequence_break_mask, "TICKET_SEQUENCE",
              "Ticket series continuity is broken; verify the sequence gap")
    df.loc[sequence_break_mask, "Ticket Sequence Check"] = "Sequence Break"
    df.loc[~sequence_break_mask & ~blank_mask(df["Ticket No"]), "Ticket Sequence Check"] = "OK"
    # Blank ticket numbers are already flagged under mandatory-field validation;
    # they are not counted as numeric sequence gaps.

    # Rule 5: Net Sale = Fare - Discount + PSF + YQ for Travel transactions.
    travel = df["DESC"].astype("string").str.strip().str.upper().eq("TRAVEL")
    fare = pd.to_numeric(df["Fare"], errors="coerce")
    discount = pd.to_numeric(df["Discount"], errors="coerce")
    psf = pd.to_numeric(df["PSF"], errors="coerce")
    yq = pd.to_numeric(df["YQ(Fuel Surcharge)"], errors="coerce")
    net_sale = pd.to_numeric(df["Net Sale"], errors="coerce")
    inputs_present = fare.notna() & discount.notna() & psf.notna() & yq.notna() & net_sale.notna()
    expected = fare - discount + psf + yq
    df.loc[travel, "Expected Net Sale"] = expected[travel]
    df.loc[travel, "Calculated Gap"] = (net_sale[travel] - expected[travel]).round(2)
    bad_travel = travel & inputs_present & ((net_sale - expected).abs() > 0.01)
    add_issue(df, bad_travel, "TRAVEL_NET_SALE",
              "Net Sale does not match Fare - Discount + PSF + YQ")

    # Cancellation/ADC comparison is intentionally excluded from this prototype indicator.

    df["Status"] = df["Issue Count"].apply(
        lambda n: "Valid / Reconciled" if n == 0 else "Requires Review"
    )
    return df


def apply_cancel_rule(df):
    df = df.copy()
    desc = df["DESC"].astype(str).str.strip().str.upper()
    travel_rows = df.loc[desc.eq("TRAVEL")].copy()
    cancel_indexes = df.index[desc.eq("CANCEL")]
    if len(cancel_indexes) == 0:
        return df
    travel_rows["_pnr_key"] = travel_rows["PNR"].astype(str).str.strip()
    travel_rows["_ticket_key"] = travel_rows["Ticket No"].astype(str).str.strip()
    travel_rows = travel_rows.sort_values("Tran Date")
    by_pnr_ticket = {key: group for key, group in travel_rows.groupby(["_pnr_key", "_ticket_key"], sort=False, dropna=False)}
    by_pnr = {key: group for key, group in travel_rows.groupby("_pnr_key", sort=False, dropna=False)}
    for idx in cancel_indexes:
        row = df.loc[idx]
        pnr, ticket = str(row["PNR"]).strip(), str(row["Ticket No"]).strip()
        candidates = by_pnr_ticket.get((pnr, ticket))
        if candidates is None or candidates.empty:
            candidates = by_pnr.get(pnr)
        if candidates is not None and not candidates.empty and pd.notna(row["Tran Date"]):
            candidates = candidates.loc[candidates["Tran Date"] <= row["Tran Date"]]
        else:
            candidates = None
        if candidates is None or candidates.empty:
            mask = pd.Series(False, index=df.index); mask.loc[idx] = True
            add_issue(df, mask, "CANCEL_MATCH", "No corresponding Travel transaction found for cancellation")
            continue
        prior = candidates.iloc[-1]
        if pd.isna(prior["Net Sale"]):
            mask = pd.Series(False, index=df.index); mask.loc[idx] = True
            add_issue(df, mask, "CANCEL_MATCH", "Corresponding Travel transaction has no Net Sale")
            continue
        hours = (row["Travel Date"] - row["Tran Date"]).total_seconds() / 3600 if pd.notna(row["Travel Date"]) and pd.notna(row["Tran Date"]) else None
        if hours is None:
            mask = pd.Series(False, index=df.index); mask.loc[idx] = True
            add_issue(df, mask, "CANCEL_TIME", "Cannot determine cancellation window")
            continue
        prior_net = float(prior["Net Sale"])
        if hours > 48:
            expected, policy = -(prior_net - 100), ">48 hours: refund less ₹100"
        elif hours > 24:
            expected, policy = -(prior_net * 0.50), "24–48 hours: 50% refund"
        else:
            expected, policy = 0.0, "Within 24 hours: no refund"
        df.loc[idx, "Expected Net Sale"] = round(expected, 2)
        df.loc[idx, "Calculated Gap"] = round(float(row["Net Sale"]) - expected, 2) if pd.notna(row["Net Sale"]) else pd.NA
        if pd.isna(row["Net Sale"]) or abs(float(row["Net Sale"]) - expected) > 0.01:
            mask = pd.Series(False, index=df.index); mask.loc[idx] = True
            add_issue(df, mask, "CANCEL_NET_SALE", f"Expected {expected:.2f} under policy: {policy}")
    return df


@st.cache_data(show_spinner=False)
def _read_report_payloads(payloads):
    frames = []
    for filename, payload in payloads:
        try:
            bio = BytesIO(payload)
            if filename.lower().endswith(".csv"):
                x = pd.read_csv(bio, low_memory=False)
                x = standardize(x)
                x["Source Report"] = filename
                frames.append(x)
            else:
                with pd.ExcelFile(bio) as xl:
                    for sheet in xl.sheet_names:
                        x = pd.read_excel(xl, sheet_name=sheet)
                        if not x.empty:
                            x = standardize(x)
                            x["Source Report"] = f"{filename} / {sheet}"
                            frames.append(x)
        except Exception as e:
            st.warning(f"Could not read {filename}: {e}")
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=FIELDS)


def load_input(files):
    return _read_report_payloads(tuple((f.name, f.getvalue()) for f in files))


def audit_event(message):
    st.session_state.setdefault("audit", [])
    st.session_state["audit"].append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Event": message})

# Keep both organization logos visible at the top of the app on every rerun/view.
APP_DIR = Path(__file__).resolve().parent
BITS_LOGO = APP_DIR / "bits_logo.png"
MAKRUZZ_LOGO = APP_DIR / "mak_logo.png"

st.markdown("""
<style>
[data-testid="stImage"] img { object-fit: contain; }
.brand-header { padding: 0.15rem 0 0.6rem 0; }
.brand-title { text-align: center; font-size: 1.45rem; font-weight: 700; line-height: 1.25; padding-top: 0.35rem; }
.brand-subtitle { text-align: center; color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)
logo_left_space, bits_logo_col, logo_gap, mak_logo_col, logo_right_space = st.columns([2, 1.2, 0.8, 1.2, 2], vertical_alignment="center")
with bits_logo_col:
    if BITS_LOGO.exists():
        st.image(str(BITS_LOGO), width=115)
with mak_logo_col:
    if MAKRUZZ_LOGO.exists():
        st.image(str(MAKRUZZ_LOGO), width=145)
st.markdown('<div class="brand-header"><div class="brand-title">Automated Data Reconciliation &amp; Analytics Framework</div><div class="brand-subtitle">Accounts Team • Report-based reconciliation prototype</div></div>', unsafe_allow_html=True)
st.divider()

if "result" not in st.session_state:
    st.session_state.result = pd.DataFrame()
if "raw_data" not in st.session_state:
    st.session_state.raw_data = pd.DataFrame(columns=FIELDS)
if "input_signature" not in st.session_state:
    st.session_state.input_signature = None

st.subheader("Report Upload & Processing")
st.caption("Upload the Master Report and run the checks directly from the main dashboard.")

with st.container(border=True):
    uploaded = st.file_uploader("Upload Master Report(s)", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="main_report_uploader")
    run = st.button("Run Reconciliation", type="primary")

if uploaded:
    signature = tuple((f.name, len(f.getvalue()), hash(f.getvalue())) for f in uploaded)
    if signature != st.session_state.input_signature:
        st.session_state.raw_data = load_input(uploaded)
        st.session_state.input_signature = signature
        st.session_state.result = pd.DataFrame()
else:
    if st.session_state.input_signature is not None:
        st.session_state.raw_data = pd.DataFrame(columns=FIELDS)
        st.session_state.result = pd.DataFrame()
        st.session_state.input_signature = None

raw = st.session_state.raw_data
if run:
    with st.spinner("Processing report and checking reconciliation rules…"):
        st.session_state.result = validate_all(raw)
    audit_event(f"Validation completed for {len(raw):,} records")
    st.success(f"Processing completed for {len(raw):,} records.")

if raw.empty and st.session_state.result.empty:
    st.info("Upload a CSV/Excel Master Report, then click **Run Reconciliation**.")
    st.stop()
if st.session_state.result.empty:
    st.info(f"Report loaded: {len(raw):,} records. Click **Run Reconciliation** to display results.")
    st.stop()
df = st.session_state.result.copy()

# Dashboard

total = len(df)
valid = int((df["Status"] == "Valid / Reconciled").sum())
review = int((df["Status"] == "Requires Review").sum())

# Key data-quality indicators requested by the Accounts Team.
required_fields = [field for field in FIELDS if field != "Order ID"]
missing_cells_by_column = blank_mask(df[required_fields])
blank_cell_count = int(missing_cells_by_column.to_numpy().sum())
rows_with_nulls = int(missing_cells_by_column.any(axis=1).sum())

# Build a cell-level exception list: one entry per missing field in each transaction.
# This makes missing Order IDs and any other blank field directly traceable to its row.
missing_cell_records = []
for row_pos, source_row in enumerate(df.index):
    missing_fields = [col for col in required_fields if bool(missing_cells_by_column.loc[source_row, col])]
    if not missing_fields:
        continue
    row = df.loc[source_row]
    for field_name in missing_fields:
        missing_cell_records.append({
            "Source Row": row.get("Source Row", row_pos + 2),
            "Sl No": row.get("Sl No", pd.NA),
            "Tran Date": row.get("Tran Date", pd.NA),
            "DESC": row.get("DESC", pd.NA),
            "PNR": row.get("PNR", pd.NA),
            "Ticket No": row.get("Ticket No", pd.NA),
            "Name": row.get("Name", pd.NA),
            "Order ID": row.get("Order ID", pd.NA),
            "Missing Field": field_name,
        })
missing_cells_df = pd.DataFrame(missing_cell_records, columns=[
    "Source Row", "Sl No", "Tran Date", "DESC", "PNR", "Ticket No", "Name", "Order ID", "Missing Field"
])
missing_by_field = (missing_cells_df.groupby("Missing Field", dropna=False).size()
                    .rename("Missing Cell Count").reset_index()
                    .sort_values("Missing Cell Count", ascending=False) if not missing_cells_df.empty
                    else pd.DataFrame(columns=["Missing Field", "Missing Cell Count"]))
duplicate_ticket_count = int((df["Ticket Duplicate Check"] == "Duplicate").sum())
duplicate_pnr_count = 0
duplicate_order_id_count = 0
exact_duplicate_transaction_count = duplicate_ticket_count
repeated_passenger_transaction_count = 0
duplicate_any_mask = df["Ticket Duplicate Check"].eq("Duplicate")
duplicate_transaction_count = int(duplicate_any_mask.sum())
_, _, missing_ticket_df, sequence_df = ticket_series_analysis(df)
missing_ticket_count = int(blank_mask(df["Ticket No"]).sum())
potential_sequence_gap_count = int(missing_ticket_df["Missing Count"].sum()) if not missing_ticket_df.empty else 0
sequence_break_count = int((df["Ticket Sequence Check"] == "Sequence Break").sum())
ticket_series_issue_count = int((blank_mask(df["Ticket No"]) | df["Ticket Sequence Check"].eq("Sequence Break")).sum())
online_mask_dashboard = (
    df["Payment Mode"].astype("string").str.strip().str.casefold().eq("online").fillna(False)
    & df["Party Name"].astype("string").str.strip().str.casefold().eq("online").fillna(False)
    & df["Location"].astype("string").str.strip().str.casefold().eq("online").fillna(False)
)
blank_online_order_id_dashboard = online_mask_dashboard & blank_mask(df["Order ID"])
net_sale_mismatch_mask = df["Issue Codes"].str.contains("TRAVEL_NET_SALE", na=False)

# Dashboard charts first: agent sales ranking and reconciliation issue profile.
st.markdown("### Dashboard Visual Summary")

dashboard_sales = df.copy()
dashboard_sales["DESC_NORMALIZED"] = dashboard_sales["DESC"].astype("string").str.strip().str.upper()
dashboard_sales["Net Sale"] = pd.to_numeric(dashboard_sales["Net Sale"], errors="coerce")
dashboard_sales["LOCATION_NORMALIZED"] = dashboard_sales["Location"].astype("string").str.strip().str.casefold()
dashboard_travel_sales = dashboard_sales[
    dashboard_sales["DESC_NORMALIZED"].eq("TRAVEL")
    & dashboard_sales["Net Sale"].notna()
    & dashboard_sales["LOCATION_NORMALIZED"].eq("agent")
].copy()
dashboard_travel_sales["Booked By"] = (
    dashboard_travel_sales["Booked By"].astype("string").fillna("Unspecified")
    .str.strip().replace("", "Unspecified")
)
top_agents = (
    dashboard_travel_sales.groupby("Booked By", as_index=False)["Net Sale"].sum()
    .sort_values("Net Sale", ascending=False).head(10)
)
issue_counts = {
    "Missing / Null Data": rows_with_nulls,
    "Duplicate Data": duplicate_transaction_count,
    "Ticket Series Break": ticket_series_issue_count,
    "Net Sale Discrepancy": int(net_sale_mismatch_mask.sum()),
    "Online Missing Order ID": int(blank_online_order_id_dashboard.sum()),
}
issue_summary = pd.DataFrame([
    {"Discrepancy": label, "Count": count,
     "Percentage": (count / total * 100) if total else 0.0}
    for label, count in issue_counts.items()
])
chart_left, chart_mid, chart_right = st.columns([1.25, 1, 1.15])

with chart_left:
    st.markdown("**Top 10 Agents by Net Sales**")
    if top_agents.empty:
        st.info("No TRAVEL sales data available for agent ranking.")
    else:
        ranked_agents = top_agents.sort_values("Net Sale", ascending=True)
        fig_agents = px.bar(
            ranked_agents, x="Net Sale", y="Booked By", orientation="h",
            text="Net Sale"
        )
        fig_agents.update_traces(texttemplate="₹%{x:,.0f}", textposition="outside")
        fig_agents.update_layout(xaxis_title="Sales (₹)", yaxis_title="", margin=dict(l=8, r=20, t=20, b=8))
        st.plotly_chart(fig_agents, use_container_width=True)

with chart_mid:
    st.markdown("**Issue Counts**")
    if issue_summary["Count"].sum() == 0:
        st.info("No discrepancies detected.")
    else:
        fig_issue_pie = px.pie(
            issue_summary[issue_summary["Count"] > 0],
            names="Discrepancy", values="Count", hole=0.35,
        )
        fig_issue_pie.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig_issue_pie, use_container_width=True)

with chart_right:
    st.markdown("**Issue Counts: Count & %**")
    ranked_issues = issue_summary.sort_values("Count", ascending=True)
    fig_issue_bar = px.bar(
        ranked_issues, x="Count", y="Discrepancy", orientation="h",
        text=ranked_issues["Percentage"].map(lambda x: f"{x:.1f}%"),
    )
    fig_issue_bar.update_layout(xaxis_title="Affected records", yaxis_title="", margin=dict(l=8, r=20, t=45, b=8))
    st.plotly_chart(fig_issue_bar, use_container_width=True)

st.markdown("### Errors/Issues Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{total:,}")
c2.metric("Transactions with Missing Values", f"{rows_with_nulls:,}", f"{(rows_with_nulls / total * 100):.1f}% of records" if total else "0%")
c3.metric("Duplicate TRAVEL Tickets", f"{duplicate_transaction_count:,}", f"{(duplicate_transaction_count / total * 100):.1f}% of records" if total else "0%")
c4.metric("Ticket Series Break / Missing Tickets", f"{ticket_series_issue_count:,}", f"{(ticket_series_issue_count / total * 100):.1f}% of records" if total else "0%")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Net Sale Discrepancies", f"{int(net_sale_mismatch_mask.sum()):,}", f"{(int(net_sale_mismatch_mask.sum()) / total * 100):.1f}% of records" if total else "0%")
c6.metric("Blank Order IDs", f"{int(blank_online_order_id_dashboard.sum()):,}", f"{(int(blank_online_order_id_dashboard.sum()) / total * 100):.1f}% of records" if total else "0%")
c7.metric("Valid / Reconciled", f"{valid:,}")
c8.metric("Requires Review", f"{review:,}")

st.divider()
# Sales analytics and lightweight projections for an at-a-glance view.
st.divider()
st.subheader("Sales Analytics")
st.caption("Sales charts use TRAVEL transactions and Net Sale values. No sales projection is displayed.")

sales_df = df.copy()
sales_df["DESC_NORMALIZED"] = sales_df["DESC"].astype("string").str.strip().str.upper()
sales_df["Tran Date"] = pd.to_datetime(sales_df["Tran Date"], errors="coerce")
sales_df["Net Sale"] = pd.to_numeric(sales_df["Net Sale"], errors="coerce")
travel_sales = sales_df[
    sales_df["DESC_NORMALIZED"].eq("TRAVEL")
    & sales_df["Tran Date"].notna()
    & sales_df["Net Sale"].notna()
].copy()

if travel_sales.empty:
    st.info("No dated TRAVEL transactions with numeric Net Sale values are available for sales analytics.")
else:
    travel_sales["Sales Date"] = travel_sales["Tran Date"].dt.normalize()
    daily_sales = (travel_sales.groupby("Sales Date", as_index=False)
                   .agg(**{"Net Sales": ("Net Sale", "sum"), "Transactions": ("Net Sale", "size")})
                   .sort_values("Sales Date"))
    sales_chart_col, mix_chart_col = st.columns([1.6, 1])

    with sales_chart_col:
        st.markdown("**Historical Daily Net Sales**")
        fig_sales = px.line(daily_sales, x="Sales Date", y="Net Sales", markers=True, title="Historical daily Net Sales")
        fig_sales.update_layout(yaxis_title="Net Sales (₹)", xaxis_title="Date")
        st.plotly_chart(fig_sales, use_container_width=True)

    with mix_chart_col:
        st.markdown("**Transaction mix**")
        mix = (sales_df["DESC_NORMALIZED"].fillna("UNSPECIFIED").value_counts()
               .rename_axis("Transaction Type").reset_index(name="Records"))
        fig_mix = px.bar(mix, x="Transaction Type", y="Records", text="Records", title="Records by transaction type")
        fig_mix.update_layout(xaxis_title="", yaxis_title="Records")
        st.plotly_chart(fig_mix, use_container_width=True)

    total_travel_sales = float(travel_sales["Net Sale"].sum())
    avg_daily_sales = float(daily_sales["Net Sales"].mean()) if not daily_sales.empty else 0.0
    k1, k2, k3 = st.columns(3)
    k1.metric("TRAVEL Net Sales", f"₹{total_travel_sales:,.2f}")
    k2.metric("Average Daily Net Sales", f"₹{avg_daily_sales:,.2f}")
    k3.metric("TRAVEL Transactions", f"{len(travel_sales):,}")
