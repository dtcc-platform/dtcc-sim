# People in DTCC

`datasets.people` returns synthetic residents, activity episodes and timed
travel stages for a representative Swedish weekday. The 2019 reference year
defines statistical inputs. The model represents a term-time weekday and uses
local clock seconds in Europe/Stockholm. It does not represent a dated
historical reconstruction. Available parameters and export capabilities are
discoverable through the DTCC Dataset registry.

## Build and play

Install the pinned Core and Sim environment with `uv sync`, then run the walking demo.

```bash
uv run python demos/dataset_people.py
```

The demo opens an animation of a small sample of synthetic walks on mapped
streets. It uses cached OpenStreetMap roads when available and otherwise
requests roads from Core. For a full population and an exportable package,
prepare the regional inputs and request the population profile:

```bash
uv run python scripts/prepare_people.py /path/to/primary johanneberg
uv run python scripts/prepare_people.py /path/to/primary malmo
export DTCC_PEOPLE_INPUTS=/path/to/primary
```

An archive contains a `2018_v3` DeSO GeoPackage and the raw SCB query/response
pairs, the SCB 1 km grid, the national 2019 RVU workbook, a regional OSM
snapshot and `region.json`. An explicitly named archive pins geographic
partitions for overlapping requests. A default live request uses a 5 km
partition with a 1 km halo. Live map queries depend on service availability.
A previously downloaded regional OSM XML extract, or a regional extract
created with `scripts/people_osm_extract.py`, can be used offline. Keep
`DTCC_PEOPLE_INPUTS` separate from evaluation files.

```python
from dtcc_sim import datasets

johanneberg = datasets.people(
    bounds=[319720, 6397660, 320220, 6398160], region="johanneberg"
)
johanneberg.export(
    "output/dataset_people/johanneberg.dtccpkg", canonical=True, format="geojson"
)

malmo = datasets.people(
    bounds=[373769, 6162013, 374269, 6162513],
    reference_year=2019,
    day_type="weekday",
    region="malmo",
    seed=42,
)
malmo.export("output/people/malmo.dtccpkg", canonical=True, format="geojson")
```

Set `transit_service_date="2019-10-15"` to use the archived schedule for
Tuesday 15 October 2019. The dataset can fetch the public archive on first use.
For an offline run, download the [14 October feed](https://data.samtrafiken.se/trafiklab/gtfs-sverige-2/2019/10/sweden-20191014.zip)
to `$DTCC_PEOPLE_INPUTS/gtfs-sverige-2-20191014.zip`. The loader verifies its
SHA-256. This date selects transit service. Activity times remain synthetic.

A package exported with `canonical=True` contains the native model.
Add `format="geojson"` to include a WGS84 playback artifact with timed
pedestrian paths and scheduled vehicle samples. The manifest identifies
the canonical model from which the playback artifact was derived.

## Sources and controls

| Input | Selection and use | Coverage and limits |
| --- | --- | --- |
| [SCB population and household tables](https://www.scb.se/hitta-statistik/regional-statistik-och-kartor/regionala-indelningar/demografiska-statistikomraden-deso/deso-tabellerna-i-ssd--information-och-instruktioner/) | `FolkmDesoAldKon`, all genders, total and 17 age bands. `HushallDesoTyp`, total and five household types. Use 2019 values on DeSO 2018. The archive stores table selections and codes. | Whole zones have register counts. Partial zones use the fraction of polygon area intersecting the regional extent. Suppressed records remain unknown and are counted separately from zeros. |
| [SCB 1 km population grid](https://www.scb.se/en/services/open-data-api/open-geodata/grid-statistics/) | `stat:befolkning_1km_2019`, `beftotalt`, reference date 31 December 2019. Use as a soft home-allocation constraint. | Disclosure protection changes margins and the grid does not identify homes. The diagnostics report residuals in entirely contained cells. Attribute Statistics Sweden under its [open data terms](https://www.scb.se/en/services/open-data-api/open-geodata/how-you-may-use-statistics-swedens-open-geodata/). |
| [Trafikanalys RVU 2019](https://www.trafa.se/transportmonster/RVU-Sverige/resvanor-i-sverige-2019-9498/) | Statistik 2020:17, table 1 annual weighted main journeys by mode and purpose, table 5 weighted journeys per person/day and 95% interval. | Swedish population ages 6–84. The defaults use published annual marginals. Linked diaries, local weekday rates and joint schedule distributions remain unavailable. The tabulated survey uncertainty travels in the model and differs from inter-seed spread. |
| [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) | Current OSM road topology, mapped crossings, building footprints and destination/stop tags. Use Core's network access rules. | ODbL 1.0. OSM data are current at archive retrieval and may depict buildings, crossings and routes absent in 2019. Polygon relations and detailed indoor networks are outside the extraction. |
| [SCB RAMS](https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__AM__AM0207__AM0207Z/PendlingKN/) | Municipal register totals for inbound, outbound and resident workers. A [workplace-based table](https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__AM__AM0207__AM0207Z/DagSni07KonKN/) tabulates workers by workplace municipality and industry. | The 2019 register lists 124,935 inbound and 53,285 outbound workers in Göteborg, and 74,041 inbound and 36,110 outbound workers in Malmö. Workplace locations and capacities use OSM and model assumptions. These register totals are context for the regional boundary, without a street-level origin–destination allocation. |
| [Samtrafiken Trafiklab GTFS Sverige 2](https://www.trafiklab.se/api/gtfs-datasets/gtfs-sverige-2/) | Optional 15 October 2019 scheduled calls from the 14 October nationwide archive. Resolve weekday calendars, date exceptions and regular pickup and drop-off. | CC0 1.0. The archive describes planned stop-area service. It supplies no observed boarding, passenger load, platform movements or vehicle geometry. |

## Synthesis and analysis

Round each DeSO total to the estimated regional coverage. Convert age and
household marginals to integer counts by largest remainder. Use child and
adult minimums by household type to assign members without replacement.
Households that cannot satisfy those constraints contribute to diagnostics.
Assign units to OSM housing footprints in proportion to remaining residential
capacity, weighted by the ratio of population grid residents to mapped
residential capacity. Downweight already-filled cells. Building capacities
assume 35 m² per person with known or estimated storeys. A missing home
capacity preserves a zonal representative point and an explicit unallocated
building type. Neither inferred occupancy nor household membership refers to
identified people.

Independent random streams are keyed by seed, zone, household, person and
stage. Choose trip modes from normalized walk, cycle, car and transit counts
in RVU table 1. Sample age-eligible work, study, education and escort tours,
plus shopping and leisure tours from the national daily marginals. Activity
times and participation use documented model assumptions in
`dtcc_sim/people/population.py`, including an 86% primary activity participation
rate and a 07:54 mean primary departure. For destination choice use weight
`sqrt(remaining capacity) × exp(-distance / mode scale)` over nearby anchors.
Jobs can be at any mapped non-residential anchor. Their capacity is the
mapped person capacity times an assumed job share: 1.0 for offices and
commercial buildings, 0.25 for shops and university buildings, 0.2 for
leisure venues and 0.15 for schools. When the 60 nearest workplaces have no
remaining jobs, the model can assign a nearby workplace with exhausted
capacity. It reserves capacity after finding a feasible round trip.
`work_capacity_overflow` counts completed assignments to full workplaces.
A walking or cycling outing without a destination in range switches to
transit.

Children aged 1–5 attend preschool with the 82% enrolment Skolverket reported
for metropolitan municipalities in autumn 2017, then the 86% daily
participation. One adult per household walks, rides or drives preschool and
school children aged up to 9 to the household's school anchor and fetches them
in the afternoon. The children share the adult's drop-off and pick-up stages
exactly, so the two appear together and are counted as two passages. Without a
feasible escort, preschool children stay home and school children travel on
their own. The diagnostics report `escorted_children` and `escort_unavailable`.

University students follow the Chalmers teaching blocks of 08–10, 10–12, 13–15
and 15–17, each ending a quarter before the hour. A student attends one to
four consecutive blocks, with the first block weighted towards the morning,
and leaves home to arrive about ten minutes before it starts. Between adjacent
blocks a student changes building within 450 m with probability 0.45. Over the
midday break 60% of students, and 35% of workers present from before 11:24
until after 13:12, walk to a café, restaurant or shop within 650 m for 22–42
minutes. RVU main journeys begin and end at home, work or school, so these
outings sit outside the published survey totals and remain assumptions.
[Chalmers block schedule](https://www.chalmers.se/en/education/your-studies/plan-and-conduct-your-studies/schedule) ·
[Skolverket preschool statistics 2017](https://www.skolverket.se/download/18.6bfaca41169863e6a65d331/1553967951737/pdf3949.pdf)

The baseline uses nearest feasible anchors and distance-only pedestrian
routing. The default penalizes steps and some centerlines of higher-class
roads. Walking speed uses a bounded 1.35 m/s normal distribution with age
adjustment. These speeds, crossing waits and schedule heuristics are
assumptions. Walking legs follow OSM node topology, with inferred short
connectors, and do not jump at mapped geometric crossings without a shared
node. Travel durations come from vertex distances and realized waits.

With a service date, trams follow mapped OSM tram rails between stop areas
and buses follow the road network, when the path is at most 2.5 times the
straight distance plus 50 m. Riders share the timed vehicle geometry. The
package exports each scheduled vehicle run that passes within 150 m of the view
with its route type and line.

Scheduled journeys search up to three stop areas within 550 m of each end and
time walking on the local network. A direct service is preferred when feasible
within the 30-minute wait limit. Otherwise the search tries one change at up to
six shared stop areas ordered by proximity to both ends. The default minimum
change is two minutes, with stop-area overrides from the feed. Ride geometry
interpolates between calls. The search omits changes between separate stop areas,
trip-specific transfer rules, midnight-spanning travel and service outside the
buffered region. A vehicle that continues after midnight is shown through its
last stop arrival before midnight. An infeasible outward transit journey falls back to car. A
failed return drops the outing. The dataset reports both outcomes separately.
With no service date selected, the default uses mapped OSM stops and a
four-minute wait with a straight ride proxy, as used for the released walking
showcase playback.

The regional model generates home residents outside the delivered view, so
the view includes incoming, outgoing and through journeys within the region.
It can show people at home separately from people present during a requested
interval. The regional boundary truncates distant demand. Routes and
residents near that boundary carry additional allocation uncertainty.
In 2019, 33.9% of 368,996 workers with a workplace in Göteborg municipality
and 38.6% of 191,766 with a workplace in Malmö municipality commuted across
the municipal boundary. Municipal counts give the scale of external demand.
They do not resolve which workers crossed the smaller simulation boundary
or which local streets they used. [SCB workplace totals](https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__AM__AM0207__AM0207Z/DagSni07KonKN/) ·
[SCB commuting totals](https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__AM__AM0207__AM0207Z/PendlingKN/)

Per-view results retain full daily paths for included people. A view boundary
filters positions, and a screenline defines its own counting geometry.
Household IDs remain resolvable through the membership index.

The timed GeoJSON derivative also carries presentation facts. Each building
has a height with its source: a mapped `height`, mapped `building:levels` at
3 m per level, or 3 m per inferred floor. Indoor intervals and waits name
their building or stop, so viewers can show occupancy. People under 13 carry
a child flag for figure size. The daylight date sets sun positions: the
service date when scheduled transit is selected, otherwise an illustrative
15 October 2019 because the representative weekday has no calendar date.

### Authorized diary templates

The public RVU workbook identifies marginal trip rates, purpose, mode and
travel durations. Linked departure times, activities within a day, and weekday
participation by joint age and purpose require linked observations or a supplied
tabulation. Trafikanalys describes its [microdata access procedure](https://www.trafa.se/sidor/utlamnande_av_mikrodata/).

To use an authorized weekday survey aggregation, put `rvu-diaries-2019.json`
in the private `DTCC_PEOPLE_INPUTS` directory and request
`survey_profile="local_diaries"`. Each row is a weighted home-to-home diary
template. Group equivalent anonymized schedules and sum their survey weights
within disjoint age cohorts. A `departure_bin` is the zero-based half-hour
from midnight. An empty `journeys` list represents a recorded no-travel day.
Purpose values are `home`, `work`, `study`, `education`, `escort`, `shopping`
and `leisure`. Main modes are `walk`, `cycle`, `car` and `transit`.
The following example uses fabricated values.

```json
{
  "reference_year": 2019,
  "day_type": "weekday",
  "source_name": "Example weighted weekday templates",
  "source_url": "https://example.test/survey",
  "cohorts": [
    {
      "age_min": 25,
      "age_max": 64,
      "diaries": [
        {
          "weight": 4.2,
          "journeys": [
            {"purpose": "work", "mode": "walk", "departure_bin": 16},
            {"purpose": "shopping", "mode": "walk", "departure_bin": 34},
            {"purpose": "home", "mode": "walk", "departure_bin": 35}
          ]
        },
        {"weight": 1.7, "journeys": []}
      ]
    }
  ]
}
```

The model samples one template by cohort and weight. It chooses departure
seconds within the bins, assigns synthetic destinations, and retains the
ordered stops, including trips between two activities away from home. A
supplied diary takes precedence over synthetic household escort scheduling. If a
chain cannot be placed or routed, the person stays at home and the failure
appears separately from survey no-travel days. Ages outside supplied cohorts
use the published-input rules. The package records the template archive hash
and source reference. It omits the input rows. The GAPSIM results apply to the
published-input profile. The diary adapter has an offline fixture and still
needs verification with an authorized observed survey extract.

Intervals use `[start, stop)`. Segment flows report simulated pedestrian
traversals per 300 seconds, assigned at each segment's midpoint time. An
oriented screenline counts a left-to-right crossing as positive. A touch
and return contributes zero. Use an edge or height selector when a line
intersects separate decks. The model uses a flat local vertical reference
and must not be used to infer heights or grade across the city. Indoor
experiences show stationary locations. Connector geometry, crossings,
entrances, access and capacity require site verification where decisions
depend on them.

The Johanneberg intervention closes a mapped link on the same realized
population and re-times affected walking stages. It is a route redistribution
experiment at fixed demand. It uses the fixed-wait scenario and requires a
feasible detour for every affected stage. It does not predict behaviour after a permanent
street change or changes in crowd interaction.

## Evidence and provenance

The native package manifest records input SHA-256 hashes, table selections,
region, bounds, dataset settings and diagnostics. The validation annotation
points to the model evaluation. GAPSIM remains evaluation evidence outside the
generation lineage.

For the scheduled option, the manifest includes the source archive hash and
the selected service date. Ride segment identifiers name the GTFS trip and
destination call sequence. The package verifier checks the exported rides
against the archive. The independent GAPSIM report covers the fixed-wait
scenario and the scheduled settings in matching neighbourhoods and seeds.
Schedule agreement is a source consistency check.

To audit the packages from the archive, run each site in a fresh process.
Omit `--transit-service-date` to audit the fixed-wait model. The verifier
blocks network connections and access to evaluation files, reloads the
canonical model, and checks scheduled rides against the source timetable.

```bash
uv run python scripts/verify_people.py /path/to/primary output/people/scheduled \
    --site johanneberg --transit-service-date 2019-10-15
uv run python scripts/verify_people.py /path/to/primary output/people/scheduled \
    --site malmo --transit-service-date 2019-10-15
```

The benchmark definition and commands are in
[the comparison protocol](../benchmarks/people/PROTOCOL.md).
See the [2019-v1 validation report](people-validation.md) for measured results. It compares
a pinned 2019 GAPSIM release with synthetic resident results for Johanneberg,
Askim and Hammarkullen. Both models can share administrative constraints.
Their agreement cannot validate individual journeys or field pedestrian
volumes.

Göteborg's [public traffic catalogue](https://goteborg.se/wps/portal?uri=gbglnk%3A202451315211954)
provides a walking section for Hisingsbron. The city's
[2019 traffic and travel report](https://goteborg.se/wps/portal?uri=gbglnk%3Agbg.page.e6edc492-508f-49f5-ab6b-d551e5105b55)
describes a 16-point 2019 campaign around Hjällbo, Hammarkullen and Angered
Centrum with hourly directional pedestrian and bicycle counts. Site
coordinates, observation intervals and the directional time series are needed
for matching. The report also describes a November 2018 city-centre Wi-Fi
campaign. Its device-based detection, dates and geography limit its value as
a comparison with a representative 2019 weekday. These sources establish
that the city measured pedestrian activity at specific places. They do not
verify the simulation's passage totals, timing or spatial deployment.
