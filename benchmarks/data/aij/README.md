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
repository. `python -m pytest tests/test_benchmarks_aij.py` checks that both
workbooks are found and read with the point counts above. File checksums are
not a reliable check: opening a workbook in a spreadsheet program can rewrite
its metadata without changing the data.
