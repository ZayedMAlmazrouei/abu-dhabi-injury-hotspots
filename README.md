# Abu Dhabi Injury Hotspots

Where injury accidents happen in Abu Dhabi Emirate, built from open government data:
Abu Dhabi Police accident reports (2022–2025) joined to the official road network, with a
daily-count forecast, a severity model, a year-over-year cause analysis and a bilingual
(English / Arabic) interactive map.

## What it answers

- Which roads carry the most injury accidents, and how that changed year to year.
- Which causes and collision types are growing (red-light running tripled 2022→2025;
  right-angle collisions +128%).
- Which roads did better or worse than the emirate-wide trend in 2025.
- How many accidents to expect on a given day in Abu Dhabi city (MAE ≈ 42/day, 8%).

## Data sources

| Source | What | Access |
|---|---|---|
| [data.abudhabi](https://data.abudhabi/opendata/) — Abu Dhabi Police | Traffic accident reports, 7 XLSX files (Abu Dhabi 2022–25, Al Ain 2022–23 & 2024–25, Al Dhafra 2022–25), ~142k rows | Public download (blocks non-browser clients) |
| [AD-SDI](https://sdi.gov.abudhabi/) — `arcgis.sdi.abudhabi.ae` | Navigable road network (names AR/EN, class, lanes, speed, route numbers), land polygons | Public ArcGIS REST, no login; licence not stated |
| [geoBoundaries](https://www.geoboundaries.org/) | UAE country and emirate boundaries (ADM0/ADM1, CC BY 4.0) | GitHub (LFS) |
| [Open-Meteo](https://open-meteo.com/) historical archive | Daily temperature, rain, wind, humidity, cloud for three grid points | Free API |

## Structure of the police data (important)

- **Injury sheet** (5,797 rows): one row per accident — date, time, city, street, collision
  type, cause, surface, weather, number injured. Record level.
- **No-injury and hit-and-run sheets** (137k rows): pre-aggregated by date × type × cause ×
  surface × weather with an `n_accidents` count. No street, no time. ~850k accidents.
- Three rows on 12 Feb 2024 hold 7,277 "hail / heavy rain" hit-and-run claims (the storm);
  excluded from modelling.
- Street labels are free text in Arabic. Only 63% of injury accidents could be placed on an
  official road (see "What the map leaves out" on the page).

## Pipeline

```
data/*.zip                      7 police XLSX files (downloaded by hand)
prepare.py                      stack 21 sheets, normalise headers, parse Excel dates, flag duplicates
                                -> out/accidents_clean.parquet, out/quality_report.md
train_models.py                 daily-count forecast (LightGBM, Poisson) with lags, calendar,
                                holidays, Ramadan and Open-Meteo weather; injury severity model
                                -> out/forecast_2025.csv, out/residuals_2025.csv
train_severity_roads.py         severity model with SDI road attributes (no gain; documented)
data/sdi_*.json                 SDI road geometry and attributes, pulled through a browser
                                (the ArcGIS host rejects server-side clients)
build_hotspot_data.py           match police street labels to SDI roads by name AND city,
                                apply route-number aliases (E10, E11, E22, E30, E45, E66...),
                                per-road and per-year stats, area markers, cause/type by year,
                                residuals against trend -> out/hotspot_data.json
site/template.html + i18n.json  the page; site/build.py inlines the data
out/hotspots.html               the finished page (works offline)
```

## Models

**Daily forecast** — LightGBM Poisson regression per region on 2022–24, tested on 2025.
Features: lags 1/7/14/28, 7- and 28-day means, weekday, day of year, Ramadan, UAE public
holidays, Eid, and daily weather. Abu Dhabi city MAE 41.5 (8.0%) vs 54.7 for a repeat-last-week
baseline; on rainy days weather cut the error from 19 to 12.

**Injury severity** (2+ injured vs 1) — LightGBM on 5.8k record-level injury accidents.
AUC 0.65–0.67. Road class, lanes and speed limit add nothing: what decides severity
(vehicles, speeds, occupants) is not in the open files. Reported honestly rather than tuned.

**What changed** — cause × year and collision type × year tables; red-light growth by road;
per-road residuals (2025 actual vs the 2022–24 mean scaled by emirate growth, 1.26).

## Matching police labels to roads

1. Normalise Arabic (strip diacritics, hamza forms, taa marbuta), drop street numbers.
2. Match against SDI `NAME_AR` **within the same city** (Abu Dhabi / Al Ain / Al Dhafra) —
   many names exist in two cities.
3. Route-number aliases for descriptive police labels: truck road → E30, Tarif–Liwa → E45,
   Al Ain–Dubai → E66, Shahama / Sheikh Zayed Rd / Salam St → E10, Abu Dhabi–Sila → E11, etc.
4. A freeway continuing into a neighbouring region with no police entries of its own is drawn
   through (E11, Hamim Road), stated in the tooltip.
5. District labels (Riyadh City, airport roads, Khalifa City bridge, Yas Mall) become area
   markers, not roads. Interior street, "undefined", checkpoints and intersection codes stay
   unmatched.

## Reproduce

```
pip install pandas pyarrow openpyxl scikit-learn lightgbm shapely
python prepare.py data/extracted
python train_models.py
python build_hotspot_data.py
cd site && python build.py
```

The SDI, boundary and weather pulls are one-off scripts run in a browser console (see the
`data/` JSON files); they are included so the build is reproducible without re-pulling.

## Known limits

- 37% of injury accidents have no mappable road.
- Police "weather" is 99% "clear"; the Open-Meteo join is what carries weather signal.
- Cause labels are the attending officer's judgement.
- SDI road data licence terms are not published; check before redistribution.
