# Kernel 2: signed machine values

`scaled_kernel_machine_correctness` in `ScaledWordPublic.lean` (namespace
`VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord`) is the integration
candidate for the complete previously accepted scaled kernel. It retains
both `EVEN_K` branches, the complete dynamic K loop, grouped coordinates,
wrapped input rows/columns, and the final output mask.

The original A/B byte storage is made explicit by low-eight-bit signed
interpretation on both branch loads. Integer dot products and every
accumulator update use finite 32-bit modular words. The independent
`exactAccumulation` specification is a sum of initial-memory signed products;
`accumulation_value` proves equality after low32 projection without a
no-overflow assumption. The fp32 scale suffix performs a real signed32-to-
binary32 rounding followed by one binary32 multiplication. The source
output tensor is int32, so the typed source materializes its implicit
float-to-signed32 store conversion. The historical mathematical root module
stores a Real and is kept only as the unchanged source bridge.

The numerical input domain is exactly that each active scaled fp32 value
has a defined finite signed32 toward-zero conversion. `expectedSpec` gives
its decoded rational value, toward-zero integer, and signed32 bounds.
No numerical restriction is placed on inactive lanes. The scale load keeps
its original mask **without an `other` value**, and the source's arbitrary
typed undefined value reaches the float arithmetic/conversion when masked.
`undefinedPolicyIndependent` proves that arbitrary results of undefined
integer conversions cannot change final memory.

The original suffix ignores the supplied output and scale strides: scale
addresses are the unwrapped output row, and output addresses are
`column + N * row`. The typed source preserves this behavior. Active-address
injectivity is proved internally from the mask, not added as an assumption.
The original configuration restrictions remain `K <= steps * BK` and
`EVEN_K = true -> K = steps * BK`. The public theorem covers every K position
and, for the even branch, proves every supplied K lane lies inside K.

The source bridge mechanically equates the full grouped prelude, the full
original K body (including both load branches), and the final six
index/mask statements. The only local width materialization is visible in
`widthAssignment`/`widths`. Integer promotion remains the `Op.intToReal`
node in the actual fp32 context selected by the fp32 scale load, with no
extra arithmetic event.

The nonempty witnesses cover both branches, signed inputs, partial M/N
boundaries and a fractional scaled result. The even branch produces fp32
13.5 and stores int32 13; the masked branch produces 10.5 and stores 10,
with a ragged block and an extra fully masked block. A signaling NaN from
an inactive no-other scale load is proved to yield an undefined conversion,
yet the output-padding sentinel remains 777 for every conversion fallback.

This establishes a concrete software bit-model contract. Native compiler
binding remains separate for byte loads, integer dot/update overflow,
fp32 rounding/flush behavior, and the defined signed32 conversion. Outside
the defined conversion domain no GPU saturation or sentinel policy is
asserted. No reassociation or approximate arithmetic law is used.
