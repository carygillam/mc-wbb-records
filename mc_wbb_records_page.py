#!/usr/bin/env python3
"""MC WBB Records Page

Run with:
    streamlit run mc_wbb_records_page.py

This page displays McPherson College women's basketball record holders for
single-game, season, and career categories by combining extracted record rows
with the available post-2009 stats datasets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SINGLE_GAME_RECORDS_CSV = ROOT_DIR / "mc_wbb_single_game_records.csv"
SEASON_RECORDS_CSV = ROOT_DIR / "mc_wbb_season_records.csv"
CAREER_RECORDS_CSV = ROOT_DIR / "mc_wbb_career_records.csv"
CAREER_CUMULATIVE_CSV = ROOT_DIR / "archived_datasets/mc_wbb_player_cumulative_career_by_years.csv"
SINGLE_GAME_STATS_CSV = ROOT_DIR / "mc_wbb_player_season_single_game_highs_wide_format.csv"
SEASON_STATS_CSV = ROOT_DIR / "mc_wbb_player_season_stats.csv"

RECORD_TYPES = ("Single Game", "Season", "Career")
DEFAULT_TOP_N = 15
MIN_TOP_N = 5
MAX_TOP_N = 50

REQUIRED_COLUMNS = ["RecordType", "Category", "Value", "Player", "Season", "Source", "Attempts"]
DISPLAY_COLUMNS = ["Rank", "Value", "Player", "Season", "Source"]
CURRENT_HOLDER_COLUMNS = ["Category", "Value", "Player", "Season", "Source"]
ALL_CANDIDATE_COLUMNS = ["Category", "Value", "Player", "Season", "Source"]
HOLDER_TABLE_COLUMN_WEIGHTS = [3.0, 2.0, 0.8, 1.1, 1.1, 1.0]


# Natural display order for the main current-record table.  Keep related box-score
# stats together so the page reads like a scorebook instead of an alphabetized list.
CATEGORY_DISPLAY_ORDER = (
    "Most Points Scored",
    "High Scoring Average",
    "Most Field Goals Made",
    "Most Field Goals Attempted",
    "Highest Field Goal Percentage",
    "High Field Goal Percentage",
    "Most 3-Pt Field Goals Made",
    "Most 3-Pt Field Goals Attempted",
    "High 3-Pt FG Percentage",
    "Most Free Throws Made",
    "Most Free Throws Attempted",
    "High Free Throw Percentage",
    "Most Rebounds",
    "High Rebound Average",
    "High Rebounding Average",
    "Most Assists",
    "Most Steals",
    "Most Blocked Shots",
    "Most Fouls",
    "Most Games Played",
    "Most PGames Played",
)
CATEGORY_ORDER_INDEX = {category: index for index, category in enumerate(CATEGORY_DISPLAY_ORDER)}
CATEGORY_GROUPS = (
    ("Scoring", ("points", "scoring average")),
    ("Field Goals", ("field goals made", "field goals attempted", "field goal percentage")),
    ("3-Point Shooting", ("3-pt field goals made", "3-pt field goals attempted", "3-pt fg percentage")),
    ("Free Throws", ("free throws made", "free throws attempted", "free throw percentage")),
    ("Rebounding", ("rebounds", "rebound average", "rebounding average")),
    ("Playmaking & Defense", ("assists", "steals", "blocked shots", "blocks", "fouls", "games played")),
)

# Percentage records should require a minimum number of attempts so a player
# cannot lead a season/career percentage category by going 1-for-1. These are
# defaults only; the sidebar lets you turn this rule off or adjust it.
PERCENTAGE_CATEGORIES = {
    "Highest Field Goal Percentage",
    "High 3-Pt FG Percentage",
    "High Free Throw Percentage",
}
PERCENTAGE_ATTEMPT_COLUMNS = {
    "Highest Field Goal Percentage": "FGA",
    "High 3-Pt FG Percentage": "3PTA",
    "High Free Throw Percentage": "FTA",
}
DEFAULT_PERCENTAGE_MINIMUMS = {
    "Season": {
        "Highest Field Goal Percentage": 100,
        "High 3-Pt FG Percentage": 50,
        "High Free Throw Percentage": 50,
    },
    "Career": {
        "Highest Field Goal Percentage": 200,
        "High 3-Pt FG Percentage": 100,
        "High Free Throw Percentage": 100,
    },
}


SINGLE_GAME_CATEGORY_MAP = {
    "Most Points Scored": "PTS",
    "Most Rebounds": "REB",
    "Most Assists": "AST",
    "Most Steals": "STL",
    "Most Blocked Shots": "BLK",
    "Most Field Goals Made": "FGM",
    "Most Field Goals Attempted": "FGA",
    "Most 3-Pt Field Goals Made": "3PM",
    "Most 3-Pt Field Goals Attempted": "3PA",
    "Most Free Throws Made": "FTM",
    "Most Free Throws Attempted": "FTA",
}

SEASON_CATEGORY_MAP = {
    "Most Points Scored": "PTS",
    "Most Rebounds": "REB",
    "Most Assists": "AST",
    "Most Steals": "STL",
    "Most Blocked Shots": "BLK",
    "Most Field Goals Made": "FGM",
    "Most Field Goals Attempted": "FGA",
    "Most 3-Pt Field Goals Made": "3PT",
    "Most 3-Pt Field Goals Attempted": "3PTA",
    "Most Free Throws Made": "FTM",
    "Most Free Throws Attempted": "FTA",
    "Most Fouls": "PF",
    "High Scoring Average": "PTS/G",
    "High Rebound Average": "REB/G",
    "High Free Throw Percentage": "FT%",
    "High 3-Pt FG Percentage": "3PT%",
    "Highest Field Goal Percentage": "FG%",
}

CAREER_STAT_ALIAS = {
    "points scored": "PTS",
    "field goals attempted": "FGA",
    "field goals made": "FGM",
    "3-pt field goals attempted": "3PTA",
    "3-pt field goals made": "3PT",
    "free throws attempted": "FTA",
    "free throws made": "FTM",
    "assists": "AST",
    "rebounds": "REB",
    "steals": "STL",
    "blocked shots": "BLK",
    "games played": "GP",
    "p games played": "GP",
    "free throw percentage": "FT%",
    "field goal percentage": "FG%",
    "rebounding average": "REB/G",
    "scoring average": "PTS/G",
    "3-pt fg percentage": "3PT%",
}


@dataclass(frozen=True)
class PercentageMinimumOptions:
    """Eligibility settings for percentage records."""

    enabled: bool
    season_fg_attempts: int
    season_3pt_attempts: int
    season_ft_attempts: int
    career_fg_attempts: int
    career_3pt_attempts: int
    career_ft_attempts: int


@dataclass(frozen=True)
class FilterOptions:
    """User-selected filters shared by the summary and category views."""

    sources: list[str]
    seasons: list[str]
    search_term: str
    top_n: int
    percentage_minimums: PercentageMinimumOptions


CATEGORY_ALIASES = {
    # Scoring average aliases. These are the same basketball stat and should
    # always appear as one category in the dashboard.
    "high scoring average": "High Scoring Average",
    "scoring average": "High Scoring Average",
    "most points per game": "High Scoring Average",
    "points per game": "High Scoring Average",
    "pts/g": "High Scoring Average",
    "ppg": "High Scoring Average",

    # Rebound average aliases.
    "high rebound average": "High Rebound Average",
    "high rebounding average": "High Rebound Average",
    "rebound average": "High Rebound Average",
    "rebounding average": "High Rebound Average",
    "most rebounds per game": "High Rebound Average",
    "rebounds per game": "High Rebound Average",
    "reb/g": "High Rebound Average",

    # Percentage label aliases.
    "high field goal percentage": "Highest Field Goal Percentage",
    "highest field goal percentage": "Highest Field Goal Percentage",
    "field goal percentage": "Highest Field Goal Percentage",
    "fg%": "Highest Field Goal Percentage",
    "high 3-pt field goal percentage": "High 3-Pt FG Percentage",
    "high 3-pt fg percentage": "High 3-Pt FG Percentage",
    "3-pt fg percentage": "High 3-Pt FG Percentage",
    "3pt%": "High 3-Pt FG Percentage",
    "high free throw percentage": "High Free Throw Percentage",
    "free throw percentage": "High Free Throw Percentage",
    "ft%": "High Free Throw Percentage",
}


def normalize_category(category: str) -> str:
    """Return the canonical display name for a records category.

    This prevents duplicate categories such as "High Scoring Average" and
    "Most Points Per Game" from being treated as separate records.
    """
    cleaned = re.sub(r"\s+", " ", str(category).strip())
    alias_key = cleaned.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    return CATEGORY_ALIASES.get(alias_key, cleaned)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return pd.DataFrame()


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def normalize_text_key(value: Any) -> str:
    """Normalize text fields so duplicate checks are resilient to spacing/case."""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_player_key(value: Any) -> str:
    """Return a consistent player key for duplicate detection.

    Handles common record-book name differences such as:
    - ``Bailey Brown``
    - ``Brown, Bailey``

    The display name is not changed. This key is only used to decide whether two
    rows are the same player record.
    """
    text = normalize_text_key(value)
    text = text.replace("’", "'").replace(".", "")

    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        text = f"{first} {last}".strip()

    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text




AGGREGATE_PLAYER_NAMES = {
    "team",
    "teams",
    "total",
    "totals",
    "team totals",
    "totals team",
    "opponent",
    "opponents",
    "opponent totals",
    "opponents totals",
    "mcpherson",
    "mcpherson college",
    "mcpherson bulldogs",
}


def is_aggregate_player_name(value: Any) -> bool:
    """Return True for team/opponent/total rows that should not count as player records."""
    name = normalize_text_key(value)
    compact_name = re.sub(r"[^a-z0-9]+", " ", name).strip()

    if compact_name in AGGREGATE_PLAYER_NAMES:
        return True

    # Catch common variants from scraped stats tables such as
    # "McPherson Team", "Team Total", "Opponent Totals", etc.
    aggregate_phrases = (
        "team total",
        "team totals",
        "total team",
        "totals team",
        "opponent total",
        "opponent totals",
        "opponents total",
        "opponents totals",
    )
    return any(phrase in compact_name for phrase in aggregate_phrases)


def remove_aggregate_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-player aggregate rows before building record candidates.

    Some scraped stat tables include team totals or opponent totals beside real
    player rows. Those totals can create impossible records, such as 447 field
    goals made for one season. Filtering here keeps the record dashboard limited
    to individual player records.
    """
    if df.empty or "Player" not in df.columns:
        return df

    cleaned = df.copy()
    player_names = cleaned["Player"].fillna("")
    return cleaned[~player_names.map(is_aggregate_player_name)].copy()


def normalize_value_key(value: Any) -> float | None:
    """Normalize numeric record values for duplicate checks."""
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return None
    return round(float(numeric_value), 6)


def prefer_current_stats_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicates while protecting the PDF record category.

    Rules:
    1. If the same record appears in both Current stats and Extracted record with
       the same category, keep Current stats.
    2. If Current stats has the same player/season/value as an Extracted record
       but under a different category, drop the Current stats row. In that case,
       the PDF/extracted category is treated as the source of truth for the
       category label. This prevents mistakes like 447 field-goal attempts being
       displayed as field goals made.
    3. Normalize player names so "Bailey Brown" and "Brown, Bailey" match.
    """
    if df.empty:
        return empty_candidates()

    deduped = df.copy()
    deduped["_record_type_key"] = deduped["RecordType"].map(normalize_text_key)
    deduped["_category_key"] = deduped["Category"].map(normalize_text_key)
    deduped["_player_key"] = deduped["Player"].map(normalize_player_key)
    deduped["_season_key"] = deduped["Season"].map(normalize_text_key)
    deduped["_value_key"] = deduped["Value"].map(normalize_value_key)

    # If a current-stat row has the same player/season/value as an extracted PDF
    # row but the category is different, keep the extracted row and remove the
    # current row. The extracted record file is the authority for category labels.
    extracted_exact_player_keys = set(
        deduped.loc[
            deduped["Source"] == "Extracted record",
            ["_record_type_key", "_player_key", "_season_key", "_value_key"],
        ].itertuples(index=False, name=None)
    )
    extracted_exact_category_keys = set(
        deduped.loc[
            deduped["Source"] == "Extracted record",
            ["_record_type_key", "_category_key", "_player_key", "_season_key", "_value_key"],
        ].itertuples(index=False, name=None)
    )

    # Extra guard for old season rows: if a PDF/extracted record has the same
    # season and value but a different category, the current stats row is likely
    # coming from a shifted/imported historical row. Example: Susan Sundahl's
    # 447 is field-goal attempts, not field goals made. This broader key catches
    # those cases even if the player name format/spelling differs slightly.
    extracted_season_value_keys = set(
        deduped.loc[
            deduped["Source"] == "Extracted record",
            ["_record_type_key", "_season_key", "_value_key"],
        ].itertuples(index=False, name=None)
    )
    extracted_season_value_category_keys = set(
        deduped.loc[
            deduped["Source"] == "Extracted record",
            ["_record_type_key", "_category_key", "_season_key", "_value_key"],
        ].itertuples(index=False, name=None)
    )

    def is_misclassified_current_row(row: pd.Series) -> bool:
        if row["Source"] != "Current stats":
            return False

        exact_player_key = (
            row["_record_type_key"],
            row["_player_key"],
            row["_season_key"],
            row["_value_key"],
        )
        exact_category_key = (
            row["_record_type_key"],
            row["_category_key"],
            row["_player_key"],
            row["_season_key"],
            row["_value_key"],
        )
        if exact_player_key in extracted_exact_player_keys and exact_category_key not in extracted_exact_category_keys:
            return True

        season_value_key = (row["_record_type_key"], row["_season_key"], row["_value_key"])
        season_value_category_key = (
            row["_record_type_key"],
            row["_category_key"],
            row["_season_key"],
            row["_value_key"],
        )
        return (
            season_value_key in extracted_season_value_keys
            and season_value_category_key not in extracted_season_value_category_keys
        )

    conflict_mask = deduped.apply(is_misclassified_current_row, axis=1)
    deduped = deduped[~conflict_mask].copy()

    # For true duplicates in the same category, keep Current stats.
    deduped["_source_priority"] = deduped["Source"].map({"Current stats": 0, "Extracted record": 1}).fillna(2)
    deduped = deduped.sort_values(
        [
            "_record_type_key",
            "_category_key",
            "_player_key",
            "_season_key",
            "_value_key",
            "_source_priority",
        ],
        ascending=[True, True, True, True, False, True],
    )
    deduped = deduped.drop_duplicates(
        subset=["_record_type_key", "_category_key", "_player_key", "_season_key", "_value_key"],
        keep="first",
    )
    deduped = deduped.drop(
        columns=[
            "_record_type_key",
            "_category_key",
            "_player_key",
            "_season_key",
            "_value_key",
            "_source_priority",
        ]
    )
    return deduped[REQUIRED_COLUMNS].reset_index(drop=True)


def parse_career_category(category: str) -> dict[str, Any] | None:
    text = normalize_category(category).lower()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()

    if text in {"most games played", "most pgames played"}:
        return {"stat_col": "GP", "years_filter": None, "use_final_row": True}

    patterns = (
        (r"most (.+?) in career$", None, True),
        (r"most (.+?) (\d+) years$", "years", False),
        (r"high (.+?) (\d+) years$", "years", False),
        (r"high (.+?) average$", None, True),
        (r"high (.+?) percentage$", None, True),
    )
    for pattern, years_group, use_final_row in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        stat_col = CAREER_STAT_ALIAS.get(match.group(1).strip())
        if stat_col:
            years_filter = int(match.group(2)) if years_group == "years" else None
            return {"stat_col": stat_col, "years_filter": years_filter, "use_final_row": use_final_row}

    return None


def append_candidate(
    rows: list[dict[str, Any]],
    record_type: str,
    category: str,
    value: Any,
    player: Any,
    season: Any,
    source: str,
    attempts: Any = pd.NA,
) -> None:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return
    attempts_value = pd.to_numeric(attempts, errors="coerce")
    if pd.isna(attempts_value):
        attempts_value = pd.NA

    rows.append(
        {
            "RecordType": record_type,
            "Category": normalize_category(category),
            "Value": float(value),
            "Player": str(player).strip(),
            "Season": str(season).strip(),
            "Source": source,
            "Attempts": attempts_value,
        }
    )


def add_current_stat_rows(
    rows: list[dict[str, Any]],
    current_df: pd.DataFrame,
    category_map: dict[str, str],
    record_type: str,
) -> None:
    if current_df.empty:
        return

    current_df = remove_aggregate_player_rows(current_df)
    if current_df.empty:
        return

    required_id_cols = ["Player", "Season"]
    if any(col not in current_df.columns for col in required_id_cols):
        return

    # Iterate row-by-row instead of using melt so percentage records can keep
    # their attempt denominator (FGA, 3PTA, or FTA) for eligibility checks.
    for _, source_row in current_df.iterrows():
        for category, stat_col in category_map.items():
            if stat_col not in current_df.columns:
                continue

            value = pd.to_numeric(source_row.get(stat_col), errors="coerce")
            if pd.isna(value) or value <= 0:
                continue

            canonical_category = normalize_category(category)
            attempts_col = PERCENTAGE_ATTEMPT_COLUMNS.get(canonical_category)
            attempts = source_row.get(attempts_col, pd.NA) if attempts_col else pd.NA

            append_candidate(
                rows,
                record_type=record_type,
                category=canonical_category,
                value=value,
                player=source_row["Player"],
                season=source_row["Season"],
                source="Current stats",
                attempts=attempts,
            )


def add_extracted_record_rows(rows: list[dict[str, Any]], records_df: pd.DataFrame, record_type: str) -> None:
    if records_df.empty:
        return

    records_df = records_df.rename(columns=lambda c: c.strip())
    records_df = remove_aggregate_player_rows(records_df)
    required_cols = {"Category", "Player", "Season", "Value"}
    if not required_cols.issubset(records_df.columns):
        missing = ", ".join(sorted(required_cols - set(records_df.columns)))
        st.warning(f"{record_type} records are missing required columns: {missing}")
        return

    for _, row in records_df.iterrows():
        append_candidate(
            rows,
            record_type=record_type,
            category=row["Category"],
            value=row["Value"],
            player=row["Player"],
            season=row["Season"],
            source="Extracted record",
        )


@st.cache_data
def load_single_game_candidates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add_current_stat_rows(rows, load_csv(SINGLE_GAME_STATS_CSV), SINGLE_GAME_CATEGORY_MAP, "Single Game")
    add_extracted_record_rows(rows, load_csv(SINGLE_GAME_RECORDS_CSV), "Single Game")
    return prefer_current_stats_duplicates(pd.DataFrame(rows, columns=REQUIRED_COLUMNS)) if rows else empty_candidates()


@st.cache_data
def load_season_candidates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add_current_stat_rows(rows, load_csv(SEASON_STATS_CSV), SEASON_CATEGORY_MAP, "Season")
    add_extracted_record_rows(rows, load_csv(SEASON_RECORDS_CSV), "Season")
    return prefer_current_stats_duplicates(pd.DataFrame(rows, columns=REQUIRED_COLUMNS)) if rows else empty_candidates()


@st.cache_data
def load_career_candidates() -> pd.DataFrame:
    records_df = load_csv(CAREER_RECORDS_CSV)
    current_df = load_csv(CAREER_CUMULATIVE_CSV)
    rows: list[dict[str, Any]] = []

    if not records_df.empty and not current_df.empty and "Category" in records_df.columns:
        current_df = current_df.rename(columns=lambda c: c.strip())
        current_df = remove_aggregate_player_rows(current_df)
        if {"Player", "Years"}.issubset(current_df.columns):
            current_df["Player"] = current_df["Player"].astype(str).str.strip()
            current_df["Seasons"] = current_df.get("Seasons", "").astype(str).str.strip()
            current_df["Years"] = pd.to_numeric(current_df["Years"], errors="coerce")
            current_df = current_df.dropna(subset=["Years"])

            if not current_df.empty:
                final_idx = current_df.groupby("Player")["Years"].idxmax()
                final_rows = current_df.loc[final_idx].copy()

                for category in sorted(records_df["Category"].dropna().unique()):
                    parsed = parse_career_category(category)
                    if not parsed:
                        continue
                    stat_col = parsed["stat_col"]
                    source_rows = final_rows if parsed["use_final_row"] else current_df[current_df["Years"] == parsed["years_filter"]]
                    if stat_col not in source_rows.columns:
                        continue

                    source_rows = source_rows.copy()
                    source_rows[stat_col] = pd.to_numeric(source_rows[stat_col], errors="coerce")
                    source_rows = source_rows.dropna(subset=[stat_col])
                    source_rows = source_rows[source_rows[stat_col] >= 0]
                    for _, row in source_rows.iterrows():
                        append_candidate(
                            rows,
                            record_type="Career",
                            category=category,
                            value=row[stat_col],
                            player=row["Player"],
                            season=row["Seasons"],
                            source="Current stats",
                            attempts=row.get(PERCENTAGE_ATTEMPT_COLUMNS.get(normalize_category(category), ""), pd.NA),
                        )

    add_extracted_record_rows(rows, records_df, "Career")
    return prefer_current_stats_duplicates(pd.DataFrame(rows, columns=REQUIRED_COLUMNS)) if rows else empty_candidates()


def load_record_candidates(record_type: str) -> pd.DataFrame:
    loaders = {
        "Single Game": load_single_game_candidates,
        "Season": load_season_candidates,
        "Career": load_career_candidates,
    }
    return loaders.get(record_type, empty_candidates)()


def apply_filters(
    df: pd.DataFrame,
    *,
    sources: list[str] | None = None,
    seasons: list[str] | None = None,
    search_term: str = "",
    category: str | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if category:
        filtered = filtered[filtered["Category"] == category]
    if sources:
        filtered = filtered[filtered["Source"].isin(sources)]
    if seasons:
        filtered = filtered[filtered["Season"].isin(seasons)]
    if search_term:
        mask = (
            filtered["Player"].str.contains(search_term, case=False, na=False)
            | filtered["Season"].str.contains(search_term, case=False, na=False)
            | filtered["Category"].str.contains(search_term, case=False, na=False)
            | filtered["Source"].str.contains(search_term, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


def minimum_for_percentage_category(record_type: str, category: str, options: PercentageMinimumOptions) -> int | None:
    """Return the attempt minimum for a percentage category, if one applies."""
    category = normalize_category(category)
    if record_type == "Season":
        if category == "Highest Field Goal Percentage":
            return options.season_fg_attempts
        if category == "High 3-Pt FG Percentage":
            return options.season_3pt_attempts
        if category == "High Free Throw Percentage":
            return options.season_ft_attempts
    if record_type == "Career":
        if category == "Highest Field Goal Percentage":
            return options.career_fg_attempts
        if category == "High 3-Pt FG Percentage":
            return options.career_3pt_attempts
        if category == "High Free Throw Percentage":
            return options.career_ft_attempts
    return None


def apply_percentage_minimums(df: pd.DataFrame, options: PercentageMinimumOptions) -> pd.DataFrame:
    """Optionally exclude current-stat percentage rows that do not meet attempts.

    Extracted record rows are kept because those records came from the official
    historical record list and usually do not include attempt denominators.
    Current-stat rows include Attempts when the source CSV has FGA/3PTA/FTA.
    """
    if df.empty or not options.enabled or "Attempts" not in df.columns:
        return df

    filtered = df.copy()
    keep_mask = pd.Series(True, index=filtered.index)

    for idx, row in filtered.iterrows():
        if row.get("Source") != "Current stats":
            continue

        minimum = minimum_for_percentage_category(str(row.get("RecordType", "")), str(row.get("Category", "")), options)
        if minimum is None:
            continue

        attempts = pd.to_numeric(row.get("Attempts"), errors="coerce")
        if pd.isna(attempts) or attempts < minimum:
            keep_mask.loc[idx] = False

    return filtered[keep_mask].copy()


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["_source_priority"] = ranked["Source"].map({"Current stats": 0, "Extracted record": 1}).fillna(2)
    ranked["_player_key"] = ranked["Player"].map(normalize_player_key)
    ranked = ranked.sort_values(
        ["Value", "Season", "_source_priority", "_player_key", "Player"],
        ascending=[False, False, True, True, True],
    )
    ranked["Rank"] = ranked["Value"].rank(method="min", ascending=False).astype(int)
    return ranked.drop(columns=["_source_priority", "_player_key"])


def top_rows_for_category(df: pd.DataFrame, category: str, top_n: int) -> pd.DataFrame:
    category_rows = df[df["Category"] == category].copy()
    if category_rows.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    ranked = add_rank(category_rows)
    ranked["_player_key"] = ranked["Player"].map(normalize_player_key)
    ranked["_value_key"] = ranked["Value"].map(normalize_value_key)
    ranked["_source_priority"] = ranked["Source"].map({"Current stats": 0, "Extracted record": 1}).fillna(2)
    ranked = ranked.sort_values(
        ["Value", "Season", "_source_priority", "_player_key", "Player"],
        ascending=[False, False, True, True, True],
    )
    ranked = ranked.drop_duplicates(subset=["_player_key", "_value_key", "Season"], keep="first")
    return ranked[DISPLAY_COLUMNS].head(top_n)


def category_display_order(category: Any) -> tuple[int, int, str]:
    """Return a stable, human-friendly sort key for record categories.

    This is intentionally phrase-based instead of alphabetic so career rows read
    like a basketball record book:

    - Field goals made: career, 2 years, 3 years, 4 years
    - Field goals attempted: career, 2 years, 3 years, 4 years
    - 3PT made before 3PT attempted
    - FT made before FT attempted

    Important: 3-point labels contain the words "field goals", so 3PT checks
    must happen before general field-goal checks.
    """
    name = normalize_category(category)
    lowered = name.lower()

    years_match = re.search(r"(\d+)\s*years?", lowered)
    # Base/career category first, then 2-year, 3-year, 4-year variants.
    years_order = int(years_match.group(1)) if years_match else 0

    ordered_phrases: tuple[tuple[str, int], ...] = (
        ("points scored", 0),
        ("scoring average", 1),
        ("points per game", 1),
        ("field goals made", 2),
        ("field goals attempted", 3),
        ("field goal percentage", 4),
        ("3-pt field goals made", 5),
        ("3-pt field goals attempted", 6),
        ("3-pt fg percentage", 7),
        ("3-pt field goal percentage", 7),
        ("free throws made", 8),
        ("free throws attempted", 9),
        ("free throw percentage", 10),
        ("rebounds", 11),
        ("rebound average", 12),
        ("rebounding average", 12),
        ("assists", 13),
        ("steals", 14),
        ("blocked shots", 15),
        ("blocks", 15),
        ("fouls", 16),
        ("games played", 17),
        ("pgames played", 17),
    )

    # 3PT must be checked before general FG because it contains "field goals".
    priority_phrases: tuple[tuple[str, int], ...] = (
        ("3-pt field goals made", 5),
        ("3-pt field goals attempted", 6),
        ("3-pt fg percentage", 7),
        ("3-pt field goal percentage", 7),
        *ordered_phrases,
    )

    for phrase, order in priority_phrases:
        if phrase in lowered:
            return (order, years_order, name)

    return (999, years_order, name)


def category_group_label(category: Any) -> str:
    """Group categories into visual sections for an easier-to-scan records page.

    3-point records are checked before Field Goals because their labels include
    "field goals" and would otherwise appear in the wrong section.
    """
    lowered = normalize_category(category).lower()
    if any(phrase in lowered for phrase in ("3-pt field goals made", "3-pt field goals attempted", "3-pt fg percentage")):
        return "3-Point Shooting"
    if any(phrase in lowered for phrase in ("field goals made", "field goals attempted", "field goal percentage")):
        return "Field Goals"
    if any(phrase in lowered for phrase in ("points", "scoring average")):
        return "Scoring"
    if any(phrase in lowered for phrase in ("free throws made", "free throws attempted", "free throw percentage")):
        return "Free Throws"
    if any(phrase in lowered for phrase in ("rebounds", "rebound average", "rebounding average")):
        return "Rebounding"
    if any(phrase in lowered for phrase in ("assists", "steals", "blocked shots", "blocks", "fouls", "games played")):
        return "Playmaking & Defense"
    return "Other Records"

def sort_by_category_display_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records by the custom stat order while preserving normal tie-breakers."""
    if df.empty or "Category" not in df.columns:
        return df
    sorted_df = df.copy()
    sort_keys = sorted_df["Category"].map(category_display_order)
    sorted_df["_category_group_order"] = sort_keys.map(lambda item: item[0])
    sorted_df["_category_variant_order"] = sort_keys.map(lambda item: item[1])
    sorted_df["_category_name_order"] = sort_keys.map(lambda item: item[2])

    secondary_columns = [col for col in ["Value", "Season", "Player"] if col in sorted_df.columns]
    sort_columns = ["_category_group_order", "_category_variant_order", "_category_name_order", *secondary_columns]
    ascending = [True, True, True, *([False, False, True][: len(secondary_columns)])]
    sorted_df = sorted_df.sort_values(sort_columns, ascending=ascending)
    return sorted_df.drop(columns=["_category_group_order", "_category_variant_order", "_category_name_order"])


def current_holders_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return one clean current-record row per category for the landing page.

    The category detail view still shows ties and the full top list.  The landing
    page intentionally picks one representative top row per category so the
    startup screen stays easy to scan.
    """
    if df.empty:
        return pd.DataFrame(columns=["Category", "Value", "Player", "Season", "Source"])

    holders = df.copy()
    holders["_source_priority"] = holders["Source"].map({"Current stats": 0, "Extracted record": 1}).fillna(2)
    holders = holders.sort_values(
        ["Category", "Value", "_source_priority", "Season", "Player"],
        ascending=[True, False, True, False, True],
    )
    holders = holders.drop_duplicates(subset=["Category"], keep="first")
    holders = holders.drop(columns=["_source_priority"])
    return sort_by_category_display_order(holders[["Category", "Value", "Player", "Season", "Source"]])


def select_category(category: str) -> None:
    st.session_state.selected_category = category


def clear_selected_category() -> None:
    st.session_state.selected_category = None


def render_sidebar(records_df: pd.DataFrame) -> tuple[str, FilterOptions]:
    st.sidebar.header("Records")
    record_type = st.sidebar.radio("Record type", options=RECORD_TYPES, index=0, key="record_type")

    st.sidebar.header("Filters")
    available_sources = sorted(records_df["Source"].dropna().unique())
    selected_sources = st.sidebar.multiselect(
        "Sources",
        options=available_sources,
        default=available_sources,
        help="Choose whether to include Current stats, Extracted records, or both.",
    )

    available_seasons = sorted(records_df["Season"].dropna().unique(), reverse=True)
    use_all_seasons = st.sidebar.checkbox(
        "All seasons",
        value=True,
        help="Keep this on to include historical extracted records and current stat rows. Turn it off only when you want to narrow to specific seasons.",
    )
    if use_all_seasons:
        selected_seasons = []
    else:
        selected_seasons = st.sidebar.multiselect(
            "Choose seasons",
            options=available_seasons,
            default=available_seasons,
            help="Choose seasons to narrow the records displayed.",
        )

    search_term = st.sidebar.text_input(
        "Search",
        placeholder="Player, season, category, or source",
    ).strip()

    top_n = st.sidebar.slider(
        "Rows to show per category",
        min_value=MIN_TOP_N,
        max_value=MAX_TOP_N,
        value=DEFAULT_TOP_N,
        step=1,
        help="Controls how many rows appear after opening a category.",
    )

    st.sidebar.header("Record Standards")
    percentage_minimums_enabled = st.sidebar.checkbox(
        "Require minimum attempts for percentage records",
        value=True,
        help="Prevents 1-for-1 seasons from leading FG%, 3PT%, or FT% records. Extracted official records are kept because they usually do not include attempts.",
    )

    with st.sidebar.expander("Percentage minimums", expanded=False):
        st.caption("Season records")
        season_fg_attempts = st.number_input("Season FG% minimum FGA", min_value=0, max_value=1000, value=DEFAULT_PERCENTAGE_MINIMUMS["Season"]["Highest Field Goal Percentage"], step=5)
        season_3pt_attempts = st.number_input("Season 3PT% minimum 3PA", min_value=0, max_value=1000, value=DEFAULT_PERCENTAGE_MINIMUMS["Season"]["High 3-Pt FG Percentage"], step=5)
        season_ft_attempts = st.number_input("Season FT% minimum FTA", min_value=0, max_value=1000, value=DEFAULT_PERCENTAGE_MINIMUMS["Season"]["High Free Throw Percentage"], step=5)

        st.caption("Career records")
        career_fg_attempts = st.number_input("Career FG% minimum FGA", min_value=0, max_value=3000, value=DEFAULT_PERCENTAGE_MINIMUMS["Career"]["Highest Field Goal Percentage"], step=10)
        career_3pt_attempts = st.number_input("Career 3PT% minimum 3PA", min_value=0, max_value=3000, value=DEFAULT_PERCENTAGE_MINIMUMS["Career"]["High 3-Pt FG Percentage"], step=10)
        career_ft_attempts = st.number_input("Career FT% minimum FTA", min_value=0, max_value=3000, value=DEFAULT_PERCENTAGE_MINIMUMS["Career"]["High Free Throw Percentage"], step=10)

    return record_type, FilterOptions(
        sources=selected_sources,
        seasons=selected_seasons,
        search_term=search_term,
        top_n=top_n,
        percentage_minimums=PercentageMinimumOptions(
            enabled=percentage_minimums_enabled,
            season_fg_attempts=int(season_fg_attempts),
            season_3pt_attempts=int(season_3pt_attempts),
            season_ft_attempts=int(season_ft_attempts),
            career_fg_attempts=int(career_fg_attempts),
            career_3pt_attempts=int(career_3pt_attempts),
            career_ft_attempts=int(career_ft_attempts),
        ),
    )


def render_summary_metrics(df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Candidate rows", f"{len(df):,}")
    col2.metric("Categories", f"{df['Category'].nunique():,}")
    col3.metric("Current stats rows", f"{int((df['Source'] == 'Current stats').sum()):,}")


def format_value(value: Any) -> str:
    """Format record values without unnecessary decimal places."""
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return str(value)
    return f"{numeric_value:g}"


def inject_table_styles() -> None:
    """Theme-aware CSS for a clean records dashboard.

    Important: this does not force a global background or text color. Streamlit
    stays in charge of light/dark mode, while this layer adds spacing, cards,
    borders, and McPherson accent colors that work in both themes.
    """
    st.markdown(
        """
        <style>
        :root {
            --mc-maroon: #8a1f2d;
            --mc-maroon-soft: rgba(138, 31, 45, 0.12);
            --mc-maroon-border: rgba(138, 31, 45, 0.26);
            --mc-gold: #b7791f;
            --mc-border: rgba(128, 128, 128, 0.22);
            --mc-border-strong: rgba(128, 128, 128, 0.34);
            --mc-soft-shadow: 0 10px 28px rgba(0, 0, 0, 0.08);
            --mc-row-shadow: 0 4px 14px rgba(0, 0, 0, 0.045);
            --mc-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
            --mc-card: color-mix(in srgb, var(--secondary-background-color) 92%, var(--background-color) 8%);
            --mc-card-soft: color-mix(in srgb, var(--secondary-background-color) 84%, var(--background-color) 16%);
            --mc-accent-card: color-mix(in srgb, var(--secondary-background-color) 88%, var(--mc-maroon) 12%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Keep sidebar theme-native, only improve spacing and control readability. */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }
        div[data-baseweb="tag"] {
            background-color: var(--mc-maroon-soft) !important;
            border: 1px solid var(--mc-maroon-border) !important;
            color: var(--text-color) !important;
            border-radius: 999px !important;
        }
        div[data-baseweb="tag"] span {
            color: var(--text-color) !important;
        }

        .mc-hero {
            border: 1px solid var(--mc-border);
            border-radius: 22px;
            padding: 1.45rem 1.65rem;
            margin-bottom: 1.15rem;
            background:
                radial-gradient(circle at top left, rgba(183, 121, 31, 0.13), transparent 34%),
                linear-gradient(135deg, var(--mc-accent-card), var(--mc-card));
            box-shadow: var(--mc-soft-shadow);
        }
        .mc-eyebrow {
            color: var(--mc-gold);
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .mc-title {
            color: var(--text-color);
            font-size: 2.15rem;
            line-height: 1.08;
            font-weight: 900;
            margin: 0;
            letter-spacing: -0.035em;
        }
        .mc-subtitle {
            color: var(--mc-muted);
            font-size: 1rem;
            line-height: 1.6;
            margin-top: 0.65rem;
            max-width: 820px;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--mc-border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: var(--mc-card);
            box-shadow: var(--mc-row-shadow);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--mc-muted) !important;
            font-weight: 750;
        }
        div[data-testid="stMetricValue"] {
            color: var(--text-color) !important;
        }

        .records-panel {
            border: 1px solid var(--mc-border);
            border-radius: 20px;
            padding: 1rem 1.12rem;
            background: var(--mc-card);
            box-shadow: var(--mc-soft-shadow);
            margin-top: 0.7rem;
            margin-bottom: 0.85rem;
        }
        .section-title {
            color: var(--mc-maroon);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin: 1.1rem 0 0.42rem 0.15rem;
            padding-top: 0.25rem;
        }
        .record-header {
            color: var(--mc-muted);
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            padding: 0.55rem 0.7rem 0.45rem;
        }
        .record-row {
            border: 1px solid var(--mc-border);
            border-radius: 15px;
            padding: 0.12rem 0.2rem;
            margin-bottom: 0.45rem;
            background: var(--mc-card-soft);
            box-shadow: var(--mc-row-shadow);
            transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
        }
        .record-row:hover {
            border-color: var(--mc-maroon-border);
            background: color-mix(in srgb, var(--mc-card-soft) 82%, var(--mc-maroon) 18%);
            transform: translateY(-1px);
            box-shadow: 0 9px 22px rgba(0, 0, 0, 0.10);
        }
        .record-cell {
            min-height: 2.55rem;
            padding: 0.5rem 0.7rem;
            display: flex;
            align-items: center;
            color: var(--text-color);
        }
        .record-category {
            font-weight: 850;
            letter-spacing: -0.01em;
        }
        .record-player {
            font-weight: 800;
        }
        .record-value-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 3.65rem;
            border-radius: 999px;
            padding: 0.24rem 0.72rem;
            font-weight: 900;
            color: var(--text-color);
            background: var(--mc-maroon-soft);
            border: 1px solid var(--mc-maroon-border);
        }
        .record-muted {
            color: var(--mc-muted);
            font-size: 0.92rem;
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid var(--mc-border-strong);
            background: var(--mc-card);
            color: var(--text-color);
            font-weight: 850;
            min-height: 2.4rem;
            box-shadow: var(--mc-row-shadow);
        }
        div.stButton > button:hover {
            border-color: var(--mc-maroon-border);
            color: var(--text-color);
            background: color-mix(in srgb, var(--mc-card) 82%, var(--mc-maroon) 18%);
        }
        div.stButton > button:focus:not(:active) {
            border-color: var(--mc-maroon-border);
            color: var(--text-color);
            box-shadow: 0 0 0 0.15rem rgba(138, 31, 45, 0.14);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--mc-border);
            border-radius: 16px;
            overflow: hidden;
            background: var(--mc-card);
        }


        .mobile-label {
            display: none;
        }
        .desktop-only {
            display: block;
        }
        .mobile-help {
            display: none;
        }
        .rank-row {
            border: 1px solid var(--mc-border);
            border-radius: 15px;
            padding: 0.18rem 0.25rem;
            margin-bottom: 0.45rem;
            background: var(--mc-card-soft);
            box-shadow: var(--mc-row-shadow);
        }
        .rank-header {
            color: var(--mc-muted);
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            padding: 0.55rem 0.7rem 0.45rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-top: 0.75rem;
            }
            .mc-hero {
                border-radius: 18px;
                padding: 1rem 1rem;
                margin-bottom: 0.85rem;
            }
            .mc-title {
                font-size: 1.55rem;
                line-height: 1.15;
            }
            .mc-subtitle {
                font-size: 0.92rem;
                line-height: 1.45;
            }
            .records-panel {
                border-radius: 17px;
                padding: 0.9rem;
            }
            .desktop-only,
            .record-header,
            .rank-header {
                display: none !important;
            }
            .mobile-help {
                display: block;
                color: var(--mc-muted);
                font-size: 0.88rem;
                margin: 0.35rem 0 0.75rem;
            }
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            .record-row,
            .rank-row {
                border-radius: 18px;
                padding: 0.55rem 0.65rem;
                margin-bottom: 0.85rem;
                background: var(--mc-card);
            }
            .record-cell {
                min-height: auto;
                padding: 0.34rem 0.1rem;
                display: block;
            }
            .mobile-label {
                display: block;
                color: var(--mc-muted);
                font-size: 0.7rem;
                font-weight: 900;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.08rem;
            }
            .record-category {
                font-size: 1.03rem;
                line-height: 1.25;
            }
            .record-player {
                font-size: 1rem;
            }
            .record-value-pill {
                min-width: auto;
                padding: 0.22rem 0.7rem;
            }
            div.stButton > button {
                min-height: 2.75rem;
                margin-top: 0.35rem;
                font-size: 0.95rem;
            }
            div[data-testid="stMetric"] {
                padding: 0.75rem 0.85rem;
                border-radius: 15px;
            }
            div[data-testid="stDataFrame"] {
                font-size: 0.84rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_table_cell(column: Any, value: Any, css_class: str = "", label: str = "") -> None:
    label_html = f'<span class="mobile-label">{label}</span>' if label else ""
    column.markdown(
        f'<div class="record-cell {css_class}">{label_html}<span>{value}</span></div>',
        unsafe_allow_html=True,
    )


def render_holder_table_header() -> None:
    headers = ["Category", "Record Holder", "Value", "Season", "Source", ""]
    for col, label in zip(st.columns(HOLDER_TABLE_COLUMN_WEIGHTS), headers):
        col.markdown(f'<div class="record-header">{label}</div>', unsafe_allow_html=True)


def render_holder_table_row(row: pd.Series, top_n: int, index: int) -> None:
    category = str(row["Category"])
    player = str(row["Player"])
    season = str(row["Season"])
    source = str(row["Source"])
    value = format_value(row["Value"])

    st.markdown('<div class="record-row">', unsafe_allow_html=True)
    category_col, player_col, value_col, season_col, source_col, action_col = st.columns(HOLDER_TABLE_COLUMN_WEIGHTS)
    render_table_cell(category_col, category, "record-category", "Category")
    render_table_cell(player_col, player, "record-player", "Record holder")
    render_table_cell(value_col, f'<span class="record-value-pill">{value}</span>', label="Value")
    render_table_cell(season_col, season, "record-muted", "Season")
    render_table_cell(source_col, source, "record-muted", "Source")
    action_col.button(
        "View top",
        key=f"open_holder_{index}_{category}_{player}_{season}",
        on_click=select_category,
        args=(category,),
        use_container_width=True,
        help=f"Show the top {top_n} players for {category}",
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_current_holders_screen(filtered_df: pd.DataFrame, top_n: int) -> None:
    holders = current_holders_by_category(filtered_df)
    if holders.empty:
        st.info("No current record holders match the current filters.")
        return

    st.markdown(
        f"""
        <div class="records-panel">
            <div class="mc-eyebrow">Current record holders</div>
            <div class="mc-subtitle">Showing one top record holder for each of the {holders['Category'].nunique():,} categories. Use <b>View top</b> to open the ranked list for that category.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mobile-help">Each card shows the current holder. Tap <b>View top</b> to open that category ranking.</div>', unsafe_allow_html=True)

    holders = sort_by_category_display_order(holders)
    render_holder_table_header()

    current_group = None
    for index, (_, row) in enumerate(holders.iterrows()):
        group_name = category_group_label(row["Category"])
        if group_name != current_group:
            current_group = group_name
            st.markdown(f'<div class="section-title">{group_name}</div>', unsafe_allow_html=True)
        render_holder_table_row(row, top_n, index)



def render_ranked_table_header() -> None:
    headers = ["Rank", "Value", "Player", "Season", "Source"]
    for col, label in zip(st.columns([0.65, 0.9, 2.35, 1.25, 1.35]), headers):
        col.markdown(f'<div class="rank-header">{label}</div>', unsafe_allow_html=True)


def render_ranked_table_row(row: pd.Series) -> None:
    st.markdown('<div class="rank-row">', unsafe_allow_html=True)
    rank_col, value_col, player_col, season_col, source_col = st.columns([0.65, 0.9, 2.35, 1.25, 1.35])
    render_table_cell(rank_col, str(row["Rank"]), "record-player", "Rank")
    render_table_cell(value_col, f'<span class="record-value-pill">{format_value(row["Value"])}</span>', label="Value")
    render_table_cell(player_col, str(row["Player"]), "record-player", "Player")
    render_table_cell(season_col, str(row["Season"]), "record-muted", "Season")
    render_table_cell(source_col, str(row["Source"]), "record-muted", "Source")
    st.markdown('</div>', unsafe_allow_html=True)


def render_ranked_rows(top_rows: pd.DataFrame) -> None:
    render_ranked_table_header()
    for _, row in top_rows.iterrows():
        render_ranked_table_row(row)

def render_category_detail_screen(filtered_df: pd.DataFrame, category: str, top_n: int) -> None:
    top_rows = top_rows_for_category(filtered_df, category, top_n)

    back_col, title_col = st.columns([1, 5])
    with back_col:
        st.button("← Current record holders", on_click=clear_selected_category, use_container_width=True)
    with title_col:
        st.subheader(category)
        st.caption(f"Showing top {len(top_rows)} rows after filters")

    if top_rows.empty:
        st.info("No rows match this category with the current filters.")
        return

    render_ranked_rows(top_rows)

    with st.expander("Show this top list as a table", expanded=False):
        st.dataframe(top_rows, height=420, use_container_width=True, hide_index=True)

    with st.expander("Show all candidate rows for this category", expanded=False):
        category_rows = filtered_df[filtered_df["Category"] == category]
        category_rows = category_rows.sort_values(["Value", "Season", "Player"], ascending=[False, False, True])
        st.dataframe(category_rows[ALL_CANDIDATE_COLUMNS], height=520, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="MC WBB Records", page_icon="🏀", layout="wide")

    inject_table_styles()
    st.markdown(
        """
        <div class="mc-hero">
            <div class="mc-eyebrow">McPherson College Women's Basketball</div>
            <h1 class="mc-title">🏀 Records Dashboard</h1>
            <div class="mc-subtitle">Browse the current record holder for each category, then open a focused top-player list when you want to see the full rankings.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None

    # Load a first pass so the sidebar can be rendered with the current record type.
    record_type = st.session_state.get("record_type", RECORD_TYPES[0])
    records_df = load_record_candidates(record_type)
    record_type, filters = render_sidebar(records_df)
    records_df = load_record_candidates(record_type)

    if records_df.empty:
        st.warning("No candidate rows were found for the selected record type. Check that the required CSV datasets are available.")
        return

    filtered_df = apply_filters(
        records_df,
        sources=filters.sources,
        seasons=filters.seasons,
        search_term=filters.search_term,
    )
    filtered_df = apply_percentage_minimums(filtered_df, filters.percentage_minimums)

    # If the user switches record type or filters out the selected category, return to the current holders view.
    available_categories = set(filtered_df["Category"].dropna().unique())
    if st.session_state.selected_category not in available_categories:
        st.session_state.selected_category = None

    render_summary_metrics(filtered_df)
    st.write("")

    if st.session_state.selected_category:
        render_category_detail_screen(filtered_df, st.session_state.selected_category, filters.top_n)
    else:
        render_current_holders_screen(filtered_df, filters.top_n)


if __name__ == "__main__":
    main()