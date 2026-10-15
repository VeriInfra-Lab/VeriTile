# Independent metadata arithmetic findings

Reviewer: peer:/root/rotary_grid_recover. These are source/IR review notes, not implementation or acceptance authority. The pinned v1 producer receipt is `11bf46d3b53848860556a2852a96e8a5556bb37d55b0d083dfca658b7f914386`.

Independent frontend evidence is `native-review-receipt.json`, SHA256 `30218217496d7351a19c53c30ed3032874863f6c6c896b441c99011f1686b61e`. The exact production AST was copied; only the import-time test suffix was omitted. CUDA sm90 compilation was offline, with no GPU execution. Runtime-specialized probes reproduce integer types and constexpr values; optional divisibility specialization attributes were omitted. This affects optimization evidence, not the counterexample's active index calculation.

## Integer kinds and expression-local promotion

Installed specialization returns constexpr for Python integer 1, i32 for nonnegative integers below 2^31 other than that specialization, and i64 from 2^31 through 2^63-1. Torch ordinary nonnegative shape/stride metadata fits signed64. The source's actual output metadata is coupled: `stride_bn = M` and `stride_bm = 1`, while at least one input stride is 1. Generic mixed-type and all-i64 frontend probes isolate promotion mechanics; not every manually supplied generic signature is a reachable wrapper tuple.

The arithmetic model must retain each subexpression's own type. A final i64 offset does not imply earlier products were i64:

- With only stride_an i64, rm and rn remain i32. `rm * stride_am` computes i32, then its result is sign-extended into i64 at the addition. `rn` is sign-extended before `rn * stride_an`, which computes i64. See `mixed_stride_an_i64-config0.ttir`.
- With only M i64, grid_m is i64, but grid_n, width, group_id, group_id*8, and pid%width remain i32. The already-computed i32 group_id*8 is extended for the subtraction from grid_m; the i32 pid%width is extended for division by group_size. group_size, pid_m, pid_n, rm and rn then compute i64. See `mixed_M_i64-config0.ttir`.
- With N i64, grid_n and width are i64 and pid is extended for division/remainder by width. group_id*8, the subtraction/min, and both coordinates are then i64. grid_m can still have been computed in i32 before extension.
- `tl.arange(0,128)` initially has i32 type. It is sign-extended when added to an i64 tile origin.
- Addition/subtraction/multiplication use the selected finite integer width. TTIR uses signed division (`divsi`), signed remainder (`remsi`), signed minimum (`minsi`), and signed comparisons (`cmpi slt`). Truncation-toward-zero division/remainder behavior matters when wrapped intermediates become negative; natural-number division is justified only after a separate nonnegative/no-wrap refinement.

The source uses `pid % group_size`, not `(pid % width) % group_size`. Partial final groups rotate rows. The v1 inverse correctly accounts for `group_start % group_size`; independent enumeration covered 561 grids, including 135 rotated final groups.

## Masks and addresses

Coordinate arithmetic and mask arithmetic must be preserved even for lanes that a natural-number calculation would consider inactive: a wrapped negative coordinate satisfies the source's signed `< M` comparison. It is unsound to choose activity from the mathematical coordinate and only then evaluate native coordinates.

After determining the actual signed mask, false-mask lanes issue no input load or output store. A huge inactive pointer offset does not alone justify a source failure or exclusion. In the i32 singleton-stride probe, LLVM uses plain wrapping `mul i32` for the stride product and non-inbounds pointer arithmetic. Active accesses still require exact element width, sign extension into pointer arithmetic, uint64 address behavior, allocation ownership, permissions, and bounds.

`tt.addptr` is element-scaled; the input scale is the source dtype's byte width, the output scale is one byte. Input data_ptr already incorporates storageOffset; backing allocation bounds must still be checked against the actual pointer result. A wrap can alter the address while remaining inside storage; that is an as-written computation to model, not automatically a native rejection. Similarly, output collisions require an explicit effect/ownership treatment, not an arbitrary deterministic winner.

## Concrete omitted safe routes

For either input stride axis, construct a one-element fp32 backing tensor and `as_strided` shape `(1,1)` with strides `(1,2^31)` or `(2^31,1)`. The wrapper unit-stride assertion holds, maximum and inverse are 1, the sole launched program has the sole active coordinate `(0,0)`, and both active element offsets are zero. The non-unit stride specializes to i64. Both original config entries compile successfully. Even all raw padded element offsets fit signed64: at most `127*(2^31+1)`.

Replacing 2^31 by 2^30 yields an i32 specialization. Padded inactive products wrap, but actual active addresses remain zero. This route is excluded by v1's `inputArithmeticBound` even though the input stride itself fits i32.

V1 therefore has two facets of one scope blocker: i64 metadata is omitted, and its sufficient padded-i32 bound excludes safe source routes. `FiniteHost.lean` lines225–234 and the proposal's `compute.indices=int32`/`address_domain` text encode the exclusion. Repair must retain these routes and bind the revised typed arithmetic and hazards in the public contract and exact scope/numerical metadata; merely widening the prose domain would be insufficient.
