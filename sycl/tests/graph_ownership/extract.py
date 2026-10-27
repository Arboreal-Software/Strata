"""Extract exact SYCL graph structures and ownership operations for CPU testing."""
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text()
header=Path(sys.argv[2]).read_text()
parts=[]
for name in ('SessionGraphs','TokenGraph'):
    start=header.index('struct '+name+' {')
    end=header.index('\n};',start)+3
    parts.append(header[start:end])
for label in ('CHECKED SYCL GRAPH CAPTURE','CHECKED SYCL SESSION GRAPH RETIREMENT','CHECKED SYCL TOKEN GRAPH RETIREMENT'):
    start=source.index('// BEGIN '+label)
    end=source.index('// END '+label,start)
    parts.append(source[start:end])
Path(sys.argv[3]).write_text('\n'.join(parts))
