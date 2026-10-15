# Complete typed softmax forward success for four profiles

Both typed-total leaves pass private compilation with 179 unchanged source
inputs and unchanged predecessor-artifact hashes. Complete original source
execution produces an actual final state from arbitrary initial registers,
memory, launch IDs and floating algebra. Original/candidate pairs share this
final state for fp32/fp32/fp32, fp16/fp16/fp16, fp16/fp32/fp16 and
fp64/fp16/fp64 input/mask/destination profiles, every source flag and
absent/qk/bk masks, assuming only positive depth D.

The existing extended-real model is also an instance of this theorem.
Unknown mask strings are not covered by success; their separate state/failure
equivalence remains valid. This establishes structural interpreter success,
not native IEEE refinement or meaningful values for mismatched memory tags.
Private objects are not copied to the root cache. The exact sources and logs
are archived; ordinary root Lake/public replay remains a separate step.
