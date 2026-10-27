"""Extract the exact production transaction; this is not a whole-SYCL build."""
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
start = source.index('// BEGIN CHECKED PLANAR ADOPTION')
end = source.index('// END CHECKED PLANAR ADOPTION', start)
helpers_start = source.index('bool native_head_packed_release_checked(')
helpers_end = source.index('void native_mmvq_set_multi_exact(', helpers_start)
lookup_start = source.index('bool q6_packed_mmvq(')
lookup_end = source.index('\n}', lookup_start) + 2
gather_start = source.index('// BEGIN CHECKED PLANAR GATHER')
gather_end = source.index('// END CHECKED PLANAR GATHER', gather_start)
Path(sys.argv[2]).write_text(source[start:end] + source[gather_start:gather_end] + source[helpers_start:helpers_end] + source[lookup_start:lookup_end])
