"""
load_qualifying.py
------------------
Loads qualifying session data via FastF1 for a given year + circuit.
Returns structured DataFrames ready for chunking into the RAG pipeline.

Run locally — requires FastF1 and internet access to livetiming.formula1.com.

Usage:
    from load_qualifying import load_qualifying_session
    data = load_qualifying_session(year=2024, circuit="Monza")
"""

import fastf1
import pandas as pd
from pathlib import Path

# Enable caching so repeat runs are fast
CACHE_DIR = Path(__file__).parent.parent / "cache" / "fastf1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_qualifying_session(year: int, circuit: str) -> dict:
    """
    Load qualifying session data for a given year and circuit.

    Returns a dict with keys:
        - session_info: dict of top-level session metadata
        - driver_fastest_laps: DataFrame, one row per driver (their personal best lap)
        - all_laps: DataFrame, every lap logged in qualifying
        - weather: DataFrame, weather snapshots across the session
        - telemetry: dict mapping driver code -> DataFrame of car telemetry on their fastest lap
    """
    print(f"Loading {year} {circuit} Qualifying...")
    session = fastf1.get_session(year, circuit, "Q")
    session.load(telemetry=True, laps=True, weather=True)

    # --- Session metadata ---
    session_info = {
        "year": year,
        "circuit": circuit,
        "event_name": session.event["EventName"],
        "location": session.event["Location"],
        "country": session.event["Country"],
        "date": str(session.date),
    }

    # --- All laps ---
    laps = session.laps.copy()

    # Drop laps with no recorded time (out laps, in laps, aborted)
    laps = laps.dropna(subset=["LapTime"])

    # --- Driver fastest laps ---
    # pick_fastest() returns the single quickest timed lap per driver
    driver_fastest = (
        laps.groupby("Driver", group_keys=False)
        .apply(lambda grp: grp.pick_fastest())
        .reset_index(drop=True)
    )

    keep_cols = [
        "Driver", "DriverNumber", "Team",
        "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Compound", "TyreLife", "FreshTyre",
        "LapStartTime", "LapNumber",
        "IsPersonalBest",
    ]
    # Only keep cols that actually exist (FastF1 version differences)
    keep_cols = [c for c in keep_cols if c in driver_fastest.columns]
    driver_fastest = driver_fastest[keep_cols].sort_values("LapTime").reset_index(drop=True)
    driver_fastest["QualifyingPosition"] = driver_fastest.index + 1

    # --- Weather ---
    weather = session.weather_data.copy() if session.weather_data is not None else pd.DataFrame()

    # --- Per-driver telemetry on fastest lap ---
    telemetry = {}
    for _, row in driver_fastest.iterrows():
        driver = row["Driver"]
        try:
            lap = session.laps.pick_driver(driver).pick_fastest()
            tel = lap.get_telemetry()
            # Add distance-based sector markers
            tel["Driver"] = driver
            telemetry[driver] = tel
        except Exception as e:
            print(f"  Warning: could not load telemetry for {driver}: {e}")

    print(f"  Loaded {len(driver_fastest)} drivers, {len(laps)} total laps")
    print(f"  Telemetry loaded for: {list(telemetry.keys())}")

    # --- Drivers who participated but set no valid lap time ---
    # These are excluded from driver_fastest_laps but need a status chunk
    # so the LLM can explain what happened when they are named in a question.
    all_session_drivers = session.drivers  # all driver numbers in the session
    drivers_with_laps   = set(driver_fastest["Driver"].tolist())
    dnq_drivers = []
    for num in all_session_drivers:
        info     = session.get_driver(num)
        abbr     = info.get("Abbreviation", num)
        if abbr in drivers_with_laps:
            continue
        full_name = info.get("FullName", abbr)
        team      = info.get("TeamName", "Unknown")
        # Determine what happened: check their raw laps (including aborted)
        raw_laps  = session.laps[session.laps["DriverNumber"] == num]
        last_status = None
        if not raw_laps.empty:
            last_status = int(raw_laps.iloc[-1]["TrackStatus"]) if not pd.isna(raw_laps.iloc[-1]["TrackStatus"]) else None
        reason = "flying lap interrupted by red flag" if last_status == 125 else "did not complete a timed lap"
        dnq_drivers.append({
            "driver":    abbr,
            "full_name": full_name,
            "team":      team,
            "reason":    reason,
            "lap_count": len(raw_laps),
        })
        print(f"  DNQ driver: {abbr} ({full_name}) — {reason}")

    return {
        "session_info":       session_info,
        "driver_fastest_laps": driver_fastest,
        "all_laps":           laps,
        "weather":            weather,
        "telemetry":          telemetry,
        "dnq_drivers":        dnq_drivers,
    }


def summarize_session(data: dict) -> None:
    """Print a human-readable summary of what was loaded."""
    info = data["session_info"]
    fastest = data["driver_fastest_laps"]
    weather = data["weather"]

    print(f"\n{'='*50}")
    print(f"{info['event_name']} {info['year']} - Qualifying")
    print(f"Circuit: {info['circuit']}, {info['country']}")
    print(f"{'='*50}")

    print("\nQualifying Order (fastest laps):")
    for _, row in fastest.iterrows():
        lt = row["LapTime"]
        lt_str = str(lt).split("0 days ")[-1] if pd.notna(lt) else "N/A"
        print(f"  P{row['QualifyingPosition']:2d}  {row['Driver']}  ({row['Team']})  {lt_str}")

    if not weather.empty:
        avg_track = weather["TrackTemp"].mean() if "TrackTemp" in weather.columns else None
        avg_air = weather["AirTemp"].mean() if "AirTemp" in weather.columns else None
        print(f"\nWeather avg: Track {avg_track:.1f}°C / Air {avg_air:.1f}°C" if avg_track else "")


if __name__ == "__main__":
    # Quick test
    data = load_qualifying_session(year=2024, circuit="Monza")
    summarize_session(data)
