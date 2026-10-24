# Public sources and debug barrier aggregate

Ordinary root Lake build passed all 4,899 jobs; all 1,543 recursively captured local inputs were unchanged. Targets include VeriTile, VeriTileFull, TritonBenchGVerified, TritonBenchSpecExamples, the seven-module FPFrontendRegressions library, and every requested softmax/retention/isfinite public or proof leaf present at the start (excluding the new TypedFlatData development leaf).

The build validates the public forward softmax mathematical contracts and four-profile typed successful executions, full forward/backward retention public contracts and typed executions under the explicit sequential debug-barrier rule, and the complete typed isfinite original/candidate bit contracts and witnesses. The exact debug-barrier marker is sequentially inert; unrelated unsupported effects remain rejected by regressions.

This is local elaboration and declared axiom checking. Native admission, independent comparator acceptance, and complete benchmark-task coverage remain separately tracked. New typed-flat isfinite leaves added during this build were not targets of this aggregate; their isolated component receipt is separate.
