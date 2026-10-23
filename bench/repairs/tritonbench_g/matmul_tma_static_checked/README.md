This is a separate, explicitly guarded candidate for `matmul_tma`.

The original Python source and all 21 original task files are preserved. Its
historical `Native.Admitted` conditional results are not refuted: the exposed
Python calls lacked corresponding static checks. This candidate adds five
preallocation checks for positive power-of-two M/N, power-of-two K at least 16,
and positive power-of-two warp/CTA counts. It changes the public input domain.
It adds no device or resource guards. These necessary conditions do not assert
successful compilation on every architecture or for every resource request.

The JIT kernel is unchanged. It constructs three zero-origin block pointers,
loads both complete FP16 tiles, executes one FP32 dot, optionally converts to
FP16, and stores C. The grid is (1,1); there is no source K loop. B retains the
original (0,1) order. The recorded SM90 lowering uses ordinary loads/stores and
MMA/WGMMA, with no TMA descriptor operation.

`KernelOnly/Contract.lean` proves the bit-test checks equivalent to the explicit
static domain. `KernelOnly/Kernels.lean` reuses the entire readable DSL source
and derives execution success and exact scatter memory. `KernelOnly/Correct.lean`
proves the independent Real matrix product of original physical Tensor views,
C readback, and frame. Arbitrary Reals are interpreted at physical element starts.

`KernelOnly/FPEquiv.lean` proves exact native-profile results and actual Tensor
bytes. Finite signed32-to-64 lane/stride operations determine the A/B load and
C store addresses. Both complete input tiles are read before stores. Fixed K16
first/carried instructions preserve their FP32 live accumulator and original
FP16 output encoding; FP32 output is identity. Primitive correspondence is only
at those fixed instruction and encoding boundaries. No whole-dot, whole-lowering,
successful-execution, allocator, or returned-object premise supplies the result.

The generic source and fixed-instruction refinement are shape-parametric. The
public recorded hardware profile mapping is deliberately limited to the 12
offline compilations: 16x32x64, CTAs=1, warps=4, all eight flags; and 128 cubed,
CTAs=1 or 4, warps=4, no transpose, both output dtypes. Other static inputs retain
the mathematical/source result without a compiled native-profile claim. The
ordinary Tensor ABI supplies view laws, byte windows and output/input separation.
The source proves C values, all exterior output bytes, and A/B byte preservation.

Validation reuses 16 actual nonempty CPU-interpreter kernel calls and 12 real JIT
binder offline compilations of the unchanged kernel. Five guard rejections and
eight positive wrapper branches were checked separately. There was no GPU run,
no new numerical transformation, and no new primitive experiment. Native bits
are not equated with Real arithmetic or ascending sequential FMA.
