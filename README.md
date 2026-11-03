# Abu Dhabi Injury Hotspots

Abu Dhabi Police publish their accident reports on data.abudhabi. I took the 2022–2025 files,
matched the street names to the official road network, and put the injury accidents on a map.
There is also a daily forecast, a severity model, a cause-by-year breakdown, and the page works
in English and Arabic.

Live page: https://zayedmalmazrouei.github.io/abu-dhabi-injury-hotspots/

## What you can see

- Which roads have the most injury accidents and how that changed each year.
- Which causes are rising. Red-light running tripled between 2022 and 2025, and right-angle
  collisions went up 128%.
- Which roads did better or worse in 2025 than their own trend.
- A daily forecast for Abu Dhabi city, off by about 42 accidents a day (8%).

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
(vehicles, speeds, occupants) is not in the open files. I left it as is rather than tune it to look better.

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

## Credits

- **Abu Dhabi Police** — Traffic Accident Reports 2022–2025, published on the
  [Abu Dhabi Open Data Platform](https://data.abudhabi/opendata/) (Department of Government
  Enablement). Dataset pages: [AD 2022](https://data.abudhabi/opendata/dataset/detail?id=dfe36aab-84f5-45c8-8ed8-aeab63c794e4),
  [AD 2023](https://data.abudhabi/opendata/dataset/detail?id=c9d87c0c-96d1-4400-969a-f9518149a51e),
  [AD 2024](https://data.abudhabi/opendata/dataset/detail?id=ba71f7f5-a232-4b1c-a2ab-66a9e08fa33c),
  [AD 2025](https://data.abudhabi/opendata/dataset/detail?id=72e667a5-48f1-4f4c-a1e0-f780024d22f1),
  [Al Ain 2022–23](https://data.abudhabi/opendata/dataset/detail?id=ae103163-65c3-4d0f-87b1-7b9cafccabea),
  [Al Ain 2024–25](https://data.abudhabi/opendata/dataset/detail?id=6912b2d7-cdbd-4ae5-ba7b-266d81881230),
  [Al Dhafra 2022–25](https://data.abudhabi/opendata/dataset/detail?id=3326d226-aaa3-49b9-8c31-f1049d43b0f1).
- **Abu Dhabi Spatial Data Infrastructure (AD-SDI)**, Department of Municipalities and Transport —
  [sdi.gov.abudhabi](https://sdi.gov.abudhabi/); Navigable Roads (`Pub/AD_Navigable_Roads`) and
  Land (`OpenData/ADSDI_OpenData` layer 27) via `arcgis.sdi.abudhabi.ae`. Licence terms not
  published; used here for non-commercial analysis.
- **geoBoundaries** — UAE ADM0/ADM1, CC BY 4.0. Runfola D. et al. (2020) *geoBoundaries: A global
  database of political administrative boundaries*, PLoS ONE 15(4).
- **Open-Meteo** — Historical Weather API (ERA5 reanalysis, Copernicus Climate Change Service),
  CC BY 4.0. [open-meteo.com](https://open-meteo.com/).
- **UAE public holidays** — [u.ae](https://u.ae/en/information-and-services/public-holidays-and-religious-affairs/public-holidays).
- Fonts: Hanken Grotesk and Readex Pro via Google Fonts (OFL).

Made by Zayed Almazrouei, 2026. The cause of each accident is whatever the officer wrote down;
I kept it as published.

## Known limits

- 37% of injury accidents have no mappable road.
- Police "weather" is 99% "clear"; the Open-Meteo join is what carries weather signal.
- Cause labels are the attending officer's judgement.
- SDI road data licence terms are not published; check before redistribution.
