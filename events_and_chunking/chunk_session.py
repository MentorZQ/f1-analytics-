"""
chunk_session.py
----------------
Produces session-level chunks from OpenF1 data:
  - session_weather: narrative of conditions across the session
  - race_control_event: one chunk per flag / message / deletion

These chunks are critical for answering "why" questions:
  - "Why was the session red flagged?" -> race_control_event chunk
  - "Did the track temperature change across Q1 to Q3?" -> session_weather chunk
  - "Why was Alonso's lap deleted?" -> race_control_event chunk with deletion message

Input:  output of load_openf1_qualifying_context()
Output: list of Chunk objects
"""

import pandas as pd
from typing import Optional
from .chunk_types import Chunk, make_chunk_id, format_timedelta


def chunk_session_weather(openf1_data: dict, session_info: dict) -> Optional[Chunk]:
    """
    Summarise weather conditions across the qualifying session as a single chunk.

    Captures:
    - Average, min, max track and air temperature
    - Whether rainfall occurred and when
    - Humidity and wind
    - Track evolution narrative (did it get faster or slower?)

    A single weather chunk per session is sufficient — the LLM can reason
    about conditions from this narrative without needing per-lap weather rows.
    """
    weather: pd.DataFrame = openf1_data.get("weather", pd.DataFrame())

    if weather.empty:
        return None

    year    = session_info.get("year", "")
    circuit = session_info.get("circuit", session_info.get("location", "Unknown"))

    lines = [
        f"Weather Conditions: {circuit} {year} Qualifying",
    ]

    # Temperature
    if "track_temp" in weather.columns:
        t = weather["track_temp"].dropna()
        if not t.empty:
            start_t = t.iloc[0]
            end_t   = t.iloc[-1]
            delta   = end_t - start_t
            direction = "increased" if delta > 0.5 else "decreased" if delta < -0.5 else "remained stable"
            lines += [
                f"Track temperature: {t.mean():.1f}°C average "
                f"(min {t.min():.1f}°C, max {t.max():.1f}°C)",
                f"Track temp {direction} across the session "
                f"({start_t:.1f}°C at start → {end_t:.1f}°C at end, Δ{delta:+.1f}°C)",
            ]
            # Track evolution implication
            if delta > 1.5:
                lines.append(
                    "Track evolution note: rising track temperature typically improves "
                    "grip as rubber builds up — later Q3 laps likely had more grip than Q1."
                )
            elif delta < -1.5:
                lines.append(
                    "Track evolution note: falling track temperature can reduce grip "
                    "and tyre warm-up effectiveness."
                )

    if "air_temp" in weather.columns:
        t = weather["air_temp"].dropna()
        if not t.empty:
            lines.append(f"Air temperature: {t.mean():.1f}°C average (min {t.min():.1f}°C, max {t.max():.1f}°C)")

    # Rainfall
    if "rainfall" in weather.columns:
        rf = weather["rainfall"].dropna()
        if rf.any():
            # Find when rain started
            rain_rows = weather[weather["rainfall"] == True]
            first_rain = rain_rows["date"].iloc[0] if not rain_rows.empty else None
            lines.append(f"Rainfall: YES — detected during session")
            if first_rain:
                lines.append(f"Rain first detected at: {first_rain}")
        else:
            lines.append("Rainfall: None — dry session throughout")

    # Wind
    if "wind_speed" in weather.columns:
        ws = weather["wind_speed"].dropna()
        if not ws.empty:
            lines.append(f"Wind speed: {ws.mean():.1f} m/s average (max {ws.max():.1f} m/s)")

    if "wind_direction" in weather.columns:
        wd = weather["wind_direction"].dropna()
        if not wd.empty:
            lines.append(f"Wind direction: {wd.mode().iloc[0]:.0f}° (most common)")

    # Humidity
    if "humidity" in weather.columns:
        h = weather["humidity"].dropna()
        if not h.empty:
            lines.append(f"Relative humidity: {h.mean():.0f}% average")

    return Chunk(
        chunk_id=make_chunk_id("session_weather", circuit, year),
        chunk_type="session_weather",
        text="\n".join(lines),
        metadata={
            "circuit":           circuit,
            "year":              year,
            "rainfall":          bool(weather["rainfall"].any()) if "rainfall" in weather.columns else False,
            "avg_track_temp":    round(weather["track_temp"].mean(), 1) if "track_temp" in weather.columns else None,
            "avg_air_temp":      round(weather["air_temp"].mean(), 1) if "air_temp" in weather.columns else None,
        },
        source="openf1",
    )


def chunk_race_control_events(openf1_data: dict, session_info: dict) -> list[Chunk]:
    """
    One chunk per race control message.

    Each event is a separate chunk so retrieval can surface specific
    incidents (e.g. retrieve only chunks containing 'deleted' or 'VER').

    Chunk text is written as a narrative sentence, not raw API output,
    so the LLM gets something it can reason over directly.
    """
    rc: pd.DataFrame = openf1_data.get("race_control", pd.DataFrame())

    if rc.empty:
        return []

    year    = session_info.get("year", "")
    circuit = session_info.get("circuit", session_info.get("location", "Unknown"))
    chunks  = []

    for idx, row in rc.iterrows():
        msg      = str(row.get("message", "")).strip()
        category = str(row.get("category", "")).strip()
        flag     = str(row.get("flag", "")).strip()
        date     = row.get("date", "")
        lap_num  = row.get("lap_number", None)
        driver   = str(row.get("driver_number", "")).strip()
        scope    = str(row.get("scope", "")).strip()
        sector   = row.get("sector", None)

        if not msg or msg == "nan":
            continue

        # Build a narrative sentence from the raw fields
        time_str = str(date)[:19] if date else "unknown time"
        lap_str  = f" on lap {int(lap_num)}" if lap_num and not pd.isna(lap_num) else ""
        drv_str  = f" involving driver #{driver}" if driver and driver != "nan" else ""
        sec_str  = f" in sector {int(sector)}" if sector and not pd.isna(sector) else ""
        scope_str = f" ({scope} scope)" if scope and scope != "nan" else ""

        narrative = (
            f"Race Control Event — {circuit} {year} Qualifying\n"
            f"Time: {time_str}{lap_str}{drv_str}{sec_str}{scope_str}\n"
            f"Category: {category}\n"
            f"Flag: {flag if flag and flag != 'nan' else 'none'}\n"
            f"Message: {msg}"
        )

        # Extract metadata for filtering
        is_deletion = any(kw in msg.lower() for kw in ["deleted", "deletion", "invalid"])
        is_flag     = flag.lower() not in ("", "nan", "none", "clear")
        is_red_flag = "red" in flag.lower() or "red flag" in msg.lower()
        is_yellow   = "yellow" in flag.lower() or "sc" in msg.lower()

        chunks.append(Chunk(
            chunk_id=make_chunk_id("race_control_event", circuit, year, idx),
            chunk_type="race_control_event",
            text=narrative,
            metadata={
                "circuit":      circuit,
                "year":         year,
                "category":     category,
                "flag":         flag if flag != "nan" else None,
                "is_deletion":  is_deletion,
                "is_red_flag":  is_red_flag,
                "is_yellow":    is_yellow,
                "driver_number": driver if driver != "nan" else None,
                "sector":       int(sector) if sector and not pd.isna(sector) else None,
            },
            source="openf1",
        ))

    return chunks


def build_session_chunks(openf1_data: dict, session_info: dict) -> list[Chunk]:
    """
    Main entry point. Produces all session-level chunks.

    Returns:
        [weather_chunk, *race_control_chunks]
    """
    chunks = []

    weather_chunk = chunk_session_weather(openf1_data, session_info)
    if weather_chunk:
        chunks.append(weather_chunk)

    chunks += chunk_race_control_events(openf1_data, session_info)

    return chunks
