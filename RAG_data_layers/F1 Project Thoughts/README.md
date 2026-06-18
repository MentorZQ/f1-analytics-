# F1 RAG — Data Loaders

Three data sources, each with its own loader.

## Sources

| Loader | Source | What it gives you | Needs internet to F1 API? |
|--------|--------|-------------------|--------------------------|
| `load_qualifying.py` | FastF1 (official F1 feed) | Per-driver fastest lap, sector times, speed traps, telemetry (speed/throttle/brake/gear/DRS) | Yes |
| `load_openf1.py` | OpenF1 API | Weather, race control messages (flags, lap deletions), tyre stints | Yes |
| `load_track_data.py` | bacinger GeoJSON + TUMFTM | Circuit metadata, GPS outline, track width geometry | GitHub only (works anywhere) |

## Quick Start

```python
from data_loaders.load_qualifying import load_qualifying_session
from data_loaders.load_openf1 import load_openf1_qualifying_context
from data_loaders.load_track_data import load_all_track_data

# Load all three layers for Monza 2024 qualifying
fastf1_data = load_qualifying_session(year=2024, circuit="Monza")
openf1_data = load_openf1_qualifying_context(year=2024, circuit_short_name="monza")
track_data  = load_all_track_data("Monza")
```

## What each loader returns

### `load_qualifying_session(year, circuit)`
```
{
  session_info:         dict    # event name, date, location
  driver_fastest_laps:  DataFrame  # one row per driver, sorted by lap time
  all_laps:             DataFrame  # every lap in the session
  weather:              DataFrame  # FastF1 weather (if available)
  telemetry:            dict    # driver_code -> DataFrame of car telemetry
}
```

### `load_openf1_qualifying_context(year, circuit_short_name)`
```
{
  session_info:   dict        # OpenF1 session metadata
  weather:        DataFrame   # air/track temp, humidity, rainfall every ~30s
  race_control:   DataFrame   # flags, lap deletions, messages
  laps:           DataFrame   # lap timings cross-reference
  stints:         DataFrame   # tyre compound per driver
  drivers:        DataFrame   # name, team, number, abbreviation
}
```

### `load_all_track_data(circuit_name)`
```
{
  metadata: {
    name, location, length_m, altitude_m,
    first_gp_year, gps_coordinates  # [[lon, lat], ...]
  },
  geometry: DataFrame   # x_m, y_m, w_tr_right_m, w_tr_left_m
                        # None if circuit not in TUMFTM database
}
```

## Track geometry availability

TUMFTM high-res geometry (x/y coords + track width) is only available for:
**Monza, Silverstone, Spa, Suzuka, Barcelona/Catalunya, Zandvoort, Austin/COTA, Melbourne**

All other circuits fall back to GPS outline from bacinger (still useful for metadata and coordinate context).

## OpenF1 circuit short names

Use these for `circuit_short_name` parameter:
`monza`, `silverstone`, `spa`, `monaco`, `bahrain`, `suzuka`,
`barcelona`, `zandvoort`, `jeddah`, `miami`, `baku`, `singapore`,
`austin`, `interlagos`, `las_vegas`, `abu_dhabi`, `melbourne`

When in doubt, check: https://api.openf1.org/v1/sessions
