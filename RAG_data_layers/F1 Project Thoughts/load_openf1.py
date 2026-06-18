"""
load_openf1.py
--------------
Pulls supplementary qualifying context from the OpenF1 API:
  - Weather snapshots during the session
  - Race control messages (yellow flags, red flags, lap deletions)
  - Stint/tyre data per driver

OpenF1 is free for historical data (2023+), no auth required.
Docs: https://openf1.org/docs/

Usage:
    from load_openf1 import load_openf1_qualifying_context
    ctx = load_openf1_qualifying_context(year=2024, circuit_short_name="monza")
"""

import requests
import pandas as pd
from typing import Optional

BASE_URL = "https://api.openf1.org/v1"


def _normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add short aliases for OpenF1 weather columns used in the notebook."""
    if df.empty:
        return df

    df = df.copy()
    column_aliases = {
        "air_temperature": "air_temp",
        "track_temperature": "track_temp",
    }

    for source, alias in column_aliases.items():
        if source in df.columns and alias not in df.columns:
            df[alias] = df[source]

    return df


def _get(endpoint: str, params: dict) -> list:
    """Generic GET helper with basic error handling."""
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_session_key(year: int, circuit_short_name: str) -> Optional[int]:
    """
    Resolve the OpenF1 session_key for a qualifying session.

    circuit_short_name examples: "monza", "silverstone", "spa", "monaco"
    These match OpenF1's naming — check https://api.openf1.org/v1/sessions
    for the exact short names if a circuit isn't found.
    """
    candidates = [circuit_short_name]
    title_case = circuit_short_name.title()
    lower_case = circuit_short_name.lower()
    if title_case not in candidates:
        candidates.append(title_case)
    if lower_case not in candidates:
        candidates.append(lower_case)

    sessions = []
    resolved_circuit_name = circuit_short_name
    for candidate in candidates:
        try:
            sessions = _get("sessions", {
                "year": year,
                "circuit_short_name": candidate,
                "session_name": "Qualifying",
            })
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 404:
                sessions = []
            else:
                raise
        if sessions:
            resolved_circuit_name = candidate
            break

    if not sessions:
        raise ValueError(
            f"No qualifying session found for {circuit_short_name} {year}. "
            f"Check circuit_short_name spelling at https://api.openf1.org/v1/sessions"
        )

    session = sessions[0]
    print(f"Found session: {session['session_name']} @ {resolved_circuit_name} "
          f"({session['date_start'][:10]}) | session_key={session['session_key']}")
    return session["session_key"], session


def get_weather(session_key: int) -> pd.DataFrame:
    """
    Weather snapshots during the session.
    Fields: air_temp, track_temp, humidity, pressure, wind_speed, wind_direction, rainfall
    Sampled roughly every 30s during the session.
    """
    data = _get("weather", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    return _normalize_weather_columns(df.sort_values("date").reset_index(drop=True))


def get_race_control_messages(session_key: int) -> pd.DataFrame:
    """
    Race control messages: yellow flags, red flags, track limits violations,
    lap time deletions, VSC, etc.

    These are critical for your RAG system — a "slow lap" question often has
    a flag or deletion as the answer.
    """
    data = _get("race_control", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    return df.sort_values("date").reset_index(drop=True)


def get_laps(session_key: int) -> pd.DataFrame:
    """
    Lap timing data from OpenF1.
    Useful as a cross-reference to FastF1 lap times.
    Fields: driver_number, lap_number, lap_duration, sector times, speed traps,
            is_pit_out_lap, date_start.
    """
    data = _get("laps", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True, format="mixed")
    return df.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)


def get_stints(session_key: int) -> pd.DataFrame:
    """
    Tyre stint data: compound, lap_start, lap_end, tyre_age per driver.
    In qualifying this tells you which compound each driver used for their fastest lap.
    """
    data = _get("stints", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def get_drivers(session_key: int) -> pd.DataFrame:
    """Driver metadata: name, team, number, abbreviation, headshot URL."""
    data = _get("drivers", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def load_openf1_qualifying_context(year: int, circuit_short_name: str) -> dict:
    """
    Main entry point. Pulls all relevant OpenF1 data for a qualifying session.

    Returns a dict with:
        - session_info: raw session metadata from OpenF1
        - weather: DataFrame of weather snapshots
        - race_control: DataFrame of flags/messages
        - laps: DataFrame of all lap timings
        - stints: DataFrame of tyre data
        - drivers: DataFrame of driver metadata
    """
    print(f"\nLoading OpenF1 data: {year} {circuit_short_name} Qualifying")

    session_key, session_info = get_session_key(year, circuit_short_name)

    weather = get_weather(session_key)
    print(f"  Weather: {len(weather)} snapshots")

    race_control = get_race_control_messages(session_key)
    print(f"  Race control messages: {len(race_control)}")
    if not race_control.empty and "message" in race_control.columns:
        print("  Messages:", race_control["message"].tolist())

    laps = get_laps(session_key)
    print(f"  Laps: {len(laps)} entries across all drivers")

    stints = get_stints(session_key)
    print(f"  Stints: {len(stints)} tyre stints")

    drivers = get_drivers(session_key)
    print(f"  Drivers: {len(drivers)}")

    return {
        "session_info": session_info,
        "weather": weather,
        "race_control": race_control,
        "laps": laps,
        "stints": stints,
        "drivers": drivers,
    }


def summarize_openf1_context(ctx: dict) -> None:
    """Print useful summary of OpenF1 context."""
    weather = ctx["weather"]
    rc = ctx["race_control"]

    if not weather.empty:
        print("\nWeather Summary:")
        for col in ["air_temp", "track_temp", "humidity", "rainfall"]:
            if col in weather.columns:
                print(f"  {col}: min={weather[col].min():.1f}, max={weather[col].max():.1f}, "
                      f"mean={weather[col].mean():.1f}")

    if not rc.empty:
        print(f"\nRace Control ({len(rc)} messages):")
        for _, row in rc.iterrows():
            msg = row.get("message", "")
            cat = row.get("category", "")
            flag = row.get("flag", "")
            print(f"  [{cat}] {flag} — {msg}")


if __name__ == "__main__":
    # Example: Monza 2024
    ctx = load_openf1_qualifying_context(year=2024, circuit_short_name="monza")
    summarize_openf1_context(ctx)
