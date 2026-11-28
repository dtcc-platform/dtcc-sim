# Gothenburg 2019 cross-model comparison

Evaluation version 1 fixes Johanneberg (111), Askim (523) and Hammarkullen (606), with seeds 17, 42 and 83. Johanneberg is the showcase site. Askim and Hammarkullen represent contrasting settlement patterns. The reference is GAPSIM v0.0.2, DOI 10.5281/zenodo.13882078. Evaluation files are supplied only to the benchmark process. The generation process consumes SCB, Trafikanalys and OpenStreetMap files.

Use Göteborg's primary-area polygons to select simulated residents and audit the GAPSIM home points. Include all ages for demographic outcomes. Use ages 18–84 with an interpretable diary for cross-model travel statistics. Treat the literal string `None` and unmatched diary syntax as missing. An interpretable home-only diary is a known zero-trip day. Report the eligible, observed and missing denominators. Compare local clock times with one-hour bins, with Wasserstein distance in minutes. Keep network route durations separate from sampled diary durations.

Compare resident totals, household totals, age-group shares, the joint distribution of age group and household size, home cell counts on the SWEREF 1 km grid, home building type, home building occupancy, trip counts, purpose and mode shares, activity chains, departures, main-mode walk durations and route lengths, and coarse within-area versus external flows. Count each main trip once. Report route coverage and exact exclusions. The simpler simulation baseline uses unweighted building capacity, nearest destinations and shortest pedestrian distance. Its comparison uses seed 42. Report the three main seeds as a range. The RVU sampling intervals are separate survey statistics.

The reported metrics are effect sizes and descriptive agreement. Neither synthetic population is an individual or street-level ground truth. Absolute count validation requires a screenline with surveyed endpoints, counted directions, the matching calendar period and observation windows. If the city releases comparable interval observations, hold out complete measurement days and reserve earlier complete, comparable days for the same-time historical mean. Compare summed bias, MAE, and peak timing. Where only incompatible vintages, coarsened counts or indices exist, state the narrower evidence scope and leave street-flow accuracy unscored.

## Reproduce the comparisons

Archive the [GAPSIM v0.0.2 SQLite files](https://doi.org/10.5281/zenodo.13882078)
for neighbourhoods 111, 523 and 606 as `111.db`, `523.db` and `606.db`.
The benchmark checks each file against its frozen SHA-256. Store the
[Göteborg primary-area GeoJSON](https://services2.arcgis.com/RMXSOC2JOPM3VHaW/arcgis/rest/services/GTB_adm_prim%C3%A4romr%C3%A5den/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson)
as `gothenburg-primary-areas.geojson` in the same evidence directory.
Use `scripts/prepare_people.py` to prepare the four primary-only regions,
as described in the [People methods](../../docs/people.md).

```bash
export DTCC_PEOPLE_INPUTS=/path/to/primary
python -m benchmarks.people.compare /path/to/evaluation output/people/benchmark --workers 4
python -m benchmarks.people.compare /path/to/evaluation output/people/benchmark-scheduled \
    --workers 4 --transit-service-date 2019-10-15
```

The [model 0.2.1 input hashes](model-0.2.1-inputs.sha256) identify the
archived primary files, region definitions, reference databases and official
boundary polygons. From the directory containing `primary/` and
`evaluation/`, run `shasum -a 256 -c /path/to/model-0.2.1-inputs.sha256`
to verify the archive. Each comparison report records the model version,
seed, source hashes, reference and boundary hashes, effect sizes and
three-seed ranges.

The scheduled sensitivity run uses the same cohorts, three seeds and nearest
baseline. It adds the pinned Trafiklab archive to generation and uses scheduled
boarding, transfers and exit walks. The evaluation-only inputs stay fixed.

For independently collected counts, supply a CSV with `date`,
`start_second`, `end_second`, `direction` and `passages`. Use Stockholm
wall time, complete comparable weekday measurement periods and the sensor's
native interval. Orient the endpoints to match the published directions
(`+1`, `-1`, or `both`). Preserve site and collection provenance beside the CSV.

```bash
python -m benchmarks.people.field people.dtccpkg observations.csv observed-results.json \
    --holdout-start YYYY-MM-DD --screenline X0 Y0 X1 Y1
```

The Angered campaign is the preferred 2019 field source. The public report
identifies the study, but does not provide its complete screenline data.
Field metrics are unavailable until those records and site definitions can be
matched. The measurement baseline uses only dates before the held-out split.
