# AIJ benchmark workbooks

The [AIJ benchmarks](../../aij/README.md) read the wind-tunnel data of the
Architectural Institute of Japan (AIJ) from two Excel workbooks. Download them
from the AIJ page
[CFD guidebook: benchmark tests](https://www.aij.or.jp/jpn/publish/cfdguide/index_e.htm)
by clicking the data-set names of cases A and B, and save them in this
directory under exactly these names:

| File | Case | Contents |
| --- | --- | --- |
| `CaseA(1_1_2).xls` | A: 1:1:2 high-rise building | approach-flow profile, 66 points in the vertical centre plane, 60 points at each of z = 0.125b and 1.25b |
| `CaseB(4_4_1).xls` | B: 4:4:1 slab | approach-flow profile, 109 points in the vertical centre plane, 115 points at z = 12.5 mm |

Each workbook holds the geometry and measuring-point figures, the inflow
profile and the measured mean velocity components and turbulent kinetic
energy. Reading `.xls` files needs `xlrd`, in the `benchmarks` extra.

The page states no licence for the data, so the files are not part of this
repository. The published results were computed from files with these
SHA-256 checksums:

```text
c2fab8c1494e777381c3819cb8874a10d3f1c5595f701d3f83119d09bad74cff  CaseA(1_1_2).xls
cf40db4df01abb9c5e304d192f978aa6c0f979ccc2c263d3968716940ecd7cc6  CaseB(4_4_1).xls
```

Check with `shasum -a 256 *.xls` (macOS) or `sha256sum *.xls` (Linux).
