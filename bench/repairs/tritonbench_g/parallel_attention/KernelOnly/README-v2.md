# Joint physical backing correction

`PublicV2.lean` is the v2 review entry. It preserves every file and build object
frozen in v1 and adds the missing common physical memory relation. The v1
per-view Tensor projections alone were insufficient: O's projection framed Z
addresses, and DQ's projection framed DK/DV addresses. Those historical
endpoints remain unchanged; v2 uses the joint endpoints below.

## Public contracts and conclusions

`Contract.lean` is the standard contract entry. It re-exports `ShapeContract`
and `ScaleABI`, and defines `CommonBacking.ForwardBacking` and
`CommonBacking.BackwardBacking`. Each contract binds all actual output Tensor
ports to one initial physical `Memory` reader, with their actual region labels,
raw allocation shapes, and pairwise physical word separation. Outputs are
separate from the complete backing storage spans of the input Tensors. Input
Tensors may alias each other and may have nonzero storage offsets. These are
ordinary spatial/API bindings, with no desired output or execution premise.

`FPEquiv.forward_joint` and `FPEquiv.backward_joint` derive a successful full
source-grid execution and **one shared physical post** from its final tagged
state. Forward observes O and Z in that same post; backward observes partial
value-shard DQ/DK scratch and complete DV in that same post. All input and output
Tensor observations, the complete input-storage frame, and the global frame
outside the union of actual source writes use exactly that post.

`CommonBacking.JointBinding` records the common before/after bindings and
logical observations. `jointPost` obtains values from the proven source final
state, selecting slots only by integer addresses and the source write footprint.
No completed value array is supplied to a run. The input bindings are proved
from the actual Tensor logical readers used by the existing source entry state.
Every source store is proved inside its actual output shape and represented by
the joint post. Source ownership and physical separation imply unique physical
writers across all programs and ports. Empty B/H/T grids yield the original
physical reader unchanged.

## Proof map

- `JointBacking.lean`: one joint post, simultaneous backing/logical observations,
  full union frame, and preservation of all input backing storage addresses.
- `JointGeometry.lean`: store bounds from the existing dense/shard geometry and
  exact Tensor shapes, plus physical writer uniqueness.
- `JointInputs.lean`: all source inputs bind to the same actual physical reader.
- `JointCoverage.lean`: every source-written region/address has a physical port;
  empty-grid post identity.
- `JointEndpoints.lean`: direct composition with the previously checked complete
  forward and parent DQ/barrier/DKV native source grids.
- `FPEquivV2.lean` and `PublicV2.lean`: additive exports and public audits.

The existing Real grid results, actual ABI scale distinction, staged native
interpretation, local compiler mapping boundary, and kernel-only scope are
unchanged. This correction adds no host shard sum, normalization, allocator,
lifecycle, byte-heap or universal compiler theorem. Saved numerical and compiler
evidence is reused; only the additions are built. `packet-v2.json` records the
v1 manifest hash, verifies its pinned files and objects are unchanged, and pins
the new source/build receipts for independent review.
