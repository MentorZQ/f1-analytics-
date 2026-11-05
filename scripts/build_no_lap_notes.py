"""
For every driver who entered a qualifying session but set no time in it, work out from the OpenF1 data what happened, and save
a short plain-text note per driver and session in RAG_data_layers/session_notes/<collection>.json (free: OpenF1 needs no key).

    python scripts/build_no_lap_notes.py

The note states only what the data shows: no lap at all, laps that did not count, or a last lap that was not completed, with the
speed trace on it (hard braking, then 0 km/h and not moving again) and the red flag that followed. It never says "crashed": the
data shows that the car stopped, not why.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "RAG_data_layers" / "F1 Project Thoughts"))

import load_qualifying_openf1 as L  # noqa: E402

OUT = ROOT / "RAG_data_layers" / "session_notes"
SESSIONS = [(2024, "Monza", "f1_qualifying_2024_monza"), (2026, "Barcelona", "f1_qualifying_2026_barcelona"),
            (2026, "Melbourne", "f1_qualifying_2026_melbourne")]
Q3_CARS = 10


def hhmmss(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M:%S")


def segment_windows(race_control: list[dict]) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Q1, Q2, Q3 as (first green flag after the previous chequered flag, chequered flag)."""
    flags = sorted((pd.Timestamp(r["date"]), r.get("flag")) for r in race_control if r.get("flag") in ("GREEN", "CHEQUERED"))
    windows, previous_end = {}, None
    for n, end in enumerate([t for t, f in flags if f == "CHEQUERED"][:3], 1):
        start = next((t for t, f in flags if f == "GREEN" and (previous_end is None or t > previous_end) and t < end), None)
        if start is not None:
            windows[n] = (start, end)
        previous_end = end
    return windows


def describe_last_lap(car: pd.DataFrame, start: pd.Timestamp, red_flags: list[pd.Timestamp], segment: str) -> str:
    """What the speed trace of an unfinished lap shows, in words, from samples with columns date and speed."""
    c = car.copy()
    c["t"] = (pd.to_datetime(c["date"], utc=True, format="mixed") - start).dt.total_seconds()
    c = c[c["t"] >= 0].sort_values("t")
    if c.empty:
        return f"His last lap in {segment} (started {hhmmss(start)}) was not completed and there is no car data for it."
    top = c.loc[c["speed"].idxmax()]
    after = c[c["t"] > top["t"]]
    parts = [f"His last lap in {segment} (started {hhmmss(start)}) was not completed."]
    still = after[after["speed"] <= 2]
    if len(still):
        t_stop = float(still["t"].iloc[0])
        moved_again = bool((after[after["t"] > t_stop + 10]["speed"] > 10).any())
        before = after[after["t"] < t_stop]
        low = f", after braking hard from {float(before['speed'].max()):.0f} km/h" if len(before) else ""
        stop_at = start + pd.Timedelta(seconds=t_stop)
        parts.append(f"His car slowed to 0 km/h {t_stop:.0f} s into the lap (at {hhmmss(stop_at)}){low}"
                     + ("" if moved_again else ", and it did not move again during the lap window") + ": it stopped on track.")
        flag = next((f for f in red_flags if start <= f <= start + pd.Timedelta(seconds=240)), None)
        if flag is not None:
            parts.append(f"A red flag was shown at {hhmmss(flag)}, {abs((flag - stop_at).total_seconds()):.0f} s "
                         f"{'after' if flag >= stop_at else 'before'} it stopped.")
        parts.append("The data does not say why (a crash and a mechanical failure look the same in it).")
    else:
        parts.append(f"His speed did not fall to 0 km/h on it (top speed {float(top['speed']):.0f} km/h); he came back to the pits.")
    return " ".join(parts)


def build(year: int, circuit: str) -> dict[str, list[dict]]:
    sess = L.find_session(year, circuit)
    sk = sess["session_key"]
    drivers = {d["driver_number"]: d["name_acronym"] for d in L._get("drivers", session_key=sk)}
    results = {r["driver_number"]: r for r in L._get("session_result", session_key=sk)}
    laps = pd.DataFrame(L._get("laps", session_key=sk))
    laps["date_start"] = pd.to_datetime(laps["date_start"], utc=True, format="mixed")
    rc = L._get("race_control", session_key=sk)
    windows = segment_windows(rc)
    red = [pd.Timestamp(r["date"]) for r in rc if r.get("flag") == "RED"]
    notes: dict[str, list[dict]] = {}
    for num in drivers:
        r = results.get(num) or {}                    # a driver with no time at all is missing from the results
        code = drivers.get(num)
        times = list(r.get("duration") or []) + [None] * 3
        position = r.get("position")
        timed = max((i for i, v in enumerate(times[:3]) if v), default=-1) + 1
        reached = 3 if position is not None and position <= Q3_CARS else max(timed, 1)
        for n in range(1, reached + 1):
            if times[n - 1] or n not in windows:
                continue
            start, end = windows[n]
            mine = laps[(laps["driver_number"] == num) & (laps["date_start"] >= start) & (laps["date_start"] <= end)].sort_values("date_start")
            flying = mine[~mine["is_pit_out_lap"].fillna(False).astype(bool)]
            done = flying[flying["lap_duration"].notna()]
            seg = f"Q{n}"
            if mine.empty:
                text = f"The data records no laps for him in {seg}."
            elif len(done):
                text = f"He completed {len(done)} lap(s) in {seg} but none counted as a time (deleted, for example for track limits)."
            elif flying.empty:
                text = f"He only ran an out-lap in {seg} and no flying lap."
            else:
                last = flying.iloc[-1]
                lo = last["date_start"]
                car = pd.DataFrame(L._get("car_data", session_key=sk, driver_number=num,
                                          **{"date>": lo.isoformat(), "date<": (min(lo + pd.Timedelta(seconds=240), end)).isoformat()}))
                text = describe_last_lap(car, lo, red, seg) if len(car) else f"His last lap in {seg} was not completed (no car data)."
            notes.setdefault(code, []).append({"segment": seg, "text": text})
    return notes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for year, circuit, collection in SESSIONS:
        notes = build(year, circuit)
        (OUT / f"{collection}.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False))
        print(f"{collection}: {sum(len(v) for v in notes.values())} note(s) for {sorted(notes)}")
        for code, items in sorted(notes.items()):
            for it in items:
                print(f"  {code} {it['segment']}: {it['text']}")


if __name__ == "__main__":
    main()
