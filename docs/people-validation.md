# People evaluation 2019-v1

DTCC People 0.2.1 uses the published 2019 survey profile. The comparison uses GAPSIM v0.0.2 synthetic resident diaries from Somanath, Thuvander and Hollberg. Both populations are selected by home location in Göteborg's official primary areas. Each area has three scenario seeds (17, 42 and 83) and a nearest-destination baseline at seed 42 under each transit setting, for 24 realizations. The scheduled setting uses service on 15 October 2019. [Comparison protocol](../benchmarks/people/PROTOCOL.md) · [Archived input SHA-256 hashes](../benchmarks/people/model-0.2.1-inputs.sha256) · [GAPSIM release](https://doi.org/10.5281/zenodo.13882078) · [Data dictionary and publication](https://doi.org/10.1016/j.dib.2024.110945).

## Demography

Seed 42 results include all resident ages. The grid comparison uses 1 km SWEREF cells occupied in either population. Total variation compares the joint age and household-size distribution, which was withheld from fitting.

| Primary area | Residents DTCC / GAPSIM | Households DTCC / GAPSIM | Grid MAE (people/cell) | Baseline grid MAE | Age × household size total variation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Johanneberg | 8,176 / 8,132 | 5,262 / 5,360 | 268 | 256 | 0.156 |
| Askim | 10,974 / 10,939 | 4,519 / 4,673 | 411 | 439 | 0.383 |
| Hammarkullen | 8,665 / 8,226 | 2,892 / 3,205 | 467 | 524 | 0.574 |

Differences in boundary allocation, mapped building vintage and how residents are assigned to buildings affect grid agreement. Similar resident totals can coexist with different household compositions.

## Resident travel

Eligible residents are aged 18–84. All eligible GAPSIM residents in these areas had interpretable diaries, and none recorded a zero-trip day. This affects the trip-rate comparison. The differences below are DTCC minus GAPSIM. Seed ranges describe variation across three draws with fixed inputs and assumptions. The RVU sampling intervals, boundary uncertainty and uncertainty in building capacity and service supply are separate.

| Primary area | Eligible DTCC / GAPSIM | GAPSIM trips/adult-day | DTCC trips/adult-day, fixed wait | Trip-rate difference, seed range | DTCC zero-trip days, seed 42 | Walk share difference (pp), seed 42 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Johanneberg | 7,362 / 7,421 | 3.014 | 2.701 to 2.752 | −0.313 to −0.262 | 22.1% | 12.9 |
| Askim | 8,074 / 8,136 | 3.103 | 2.623 to 2.667 | −0.481 to −0.436 | 26.1% | 20.6 |
| Hammarkullen | 6,073 / 5,815 | 3.119 | 2.823 to 2.895 | −0.296 to −0.223 | 23.3% | 19.7 |

### Main walking trips

Routes are joined to diaries by resident, transit activity token and compatible mode. Distances and times below are Wasserstein distances between route distributions at seed 42 with fixed-wait transit. They describe the matched subset. DTCC counts transit access and egress walks separately.

| Primary area | Routed walks DTCC / GAPSIM | Distance (m) | Duration (min) | Unmatched GAPSIM main journeys, all modes |
| --- | ---: | ---: | ---: | ---: |
| Johanneberg | 6,543 / 2,220 | 99.7 | 0.56 | 16,014 |
| Askim | 7,466 / 743 | 200.9 | 3.99 | 14,402 |
| Hammarkullen | 6,330 / 540 | 399.7 | 6.32 | 13,406 |

### Transit timetable sensitivity

The same residents and reference files are used under both transit settings. Transit share differences are in percentage points. Departure-time distances are in minutes. Each pair is ordered fixed wait / scheduled.

| Primary area | Scheduled trip-rate difference, seed range | Transit share difference (pp), seed 42 | Scheduled transit share difference (pp), seed range | Departure Wasserstein (min), seed 42 | Transit access/egress stages, seed 42 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Johanneberg | −0.313 to −0.262 | −20.38 / −18.02 | −18.81 to −17.91 | 42.0 / 41.9 | 6,165 / 6,680 |
| Askim | −0.481 to −0.437 | −11.84 / −10.88 | −10.88 to −10.68 | 56.0 / 56.6 | 5,332 / 5,763 |
| Hammarkullen | −0.296 to −0.223 | −13.55 / −12.15 | −12.15 to −11.26 | 47.5 / 48.5 | 4,432 / 4,924 |

## Package and timetable checks

Each package was generated from archived primary inputs in a fresh process with network and evaluation-file access disabled. Counts are for seed 42. The views are 500 m squares and include people whose daily trajectories intersect the view. The 08:00 counts include people walking or waiting outdoors.

| View | People, fixed / scheduled | Walking or waiting at 08:00, fixed / scheduled | Scheduled rides verified | Distinct service trips |
| --- | ---: | ---: | ---: | ---: |
| Johanneberg | 6,979 / 7,568 | 64 / 81 | 6,125 | 1,122 |
| Malmö | 5,667 / 6,055 | 84 / 115 | 3,433 | 1,224 |

In Johanneberg the two transit settings yielded 7 and 10 pedestrian passages across the test screenline from 08:00 to 08:05. Every event matched the event in the smaller, overlapping package. Scheduled boarding and alighting matched active GTFS trips and stop times in both cities.

## Independent pedestrian counts

Göteborg's 2019 report identifies a campaign at 16 locations around Angered, including Hammarkullen. It describes hourly directional walking and cycling counts. The available records lack the screenline endpoints, dates, missingness and native-interval observations needed for a match. The Hisingsbron walking series begins in 2021, and the 2018 inner-city Wi-Fi campaign counts device detections in another area. Street passage accuracy, peak timing and the observed effect of a link closure remain unscored. [Göteborg traffic reports](https://goteborg.se/wps/portal?uri=gbglnk%3Agbg.page.e6edc492-508f-49f5-ab6b-d551e5105b55) · [Municipal traffic-count catalogue](https://goteborg.se/wps/portal?uri=gbglnk%3A202451315211954).
