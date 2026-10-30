import VeriTile.Triton
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  input : RegionName
  output : RegionName
  blockSize : Nat
  coordNumel : Nat
  outputNumel : Nat
  columnOffset : Nat
  outputStride : Nat

/-- Full original eleven-channel forward graph. The source FP32 profile makes
constexpr coefficient types explicit; Y07 retains the original subtraction,
and unary negation is retained instead of real-algebra reassociation. -/
def source (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) : ComputeKernel := triton {
  tl.region coord_ptr = tl.float32, output_ptr = tl.float32
  coord_stride = $(3)
  block_id = tl.program_id(0)
  coord_striding = tl.arange(0, $(block_size)) * coord_stride
  coord_row_offset = coord_striding + $(block_size) * coord_stride * block_id
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  y = tl.load(coord_ptr + coord_row_offset + $(1),
    mask=coord_row_offset + $(1) < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2),
    mask=coord_row_offset + $(2) < $(coord_numel))
  CONST000 = tl.full([], 1.73430461568895, dtype=tl.float32)
  CONST001 = tl.full([], 2.32681380862329, dtype=tl.float32)
  CONST002 = tl.full([], 1.60565407233314, dtype=tl.float32)
  CONST003 = tl.full([], 3.21130814466628, dtype=tl.float32)
  CONST004 = tl.full([], 3.31662479035540, dtype=tl.float32)
  CONST005 = tl.full([], 6.21867148191637, dtype=tl.float32)
  CONST006 = tl.full([], 6.21867148191637, dtype=tl.float32)
  CONST007 = tl.full([], 1.60565407233314, dtype=tl.float32)
  CONST009 = tl.full([], 11.6340690431164, dtype=tl.float32)
  CONST010 = tl.full([], 12.8452325786651, dtype=tl.float32)
  CONST011 = tl.full([], 12.4373429638327, dtype=tl.float32)
  CONST012 = tl.full([], 12.8452325786651, dtype=tl.float32)
  CONST013 = tl.full([], 13.8744369255116, dtype=tl.float32)
  CONST017 = tl.full([], 33.9852909359329, dtype=tl.float32)
  CONST018 = tl.full([], 7.35803132638072, dtype=tl.float32)
  CONST020 = tl.full([], -44.1481879582843, dtype=tl.float32)
  CONST021 = tl.full([], -41.6233107765348, dtype=tl.float32)
  CONST022 = tl.full([], -29.4321253055229, dtype=tl.float32)
  CONST023 = tl.full([], -23.2681380862329, dtype=tl.float32)
  CONST024 = tl.full([], -19.2678488679977, dtype=tl.float32)
  CONST025 = tl.full([], -19.2678488679977, dtype=tl.float32)
  CONST026 = tl.full([], -16.9926454679664, dtype=tl.float32)
  CONST027 = tl.full([], -16.9926454679664, dtype=tl.float32)
  CONST028 = tl.full([], -13.8744369255116, dtype=tl.float32)
  CONST029 = tl.full([], -16.5831239517770, dtype=tl.float32)
  CONST030 = tl.full([], 3.46860923137790, dtype=tl.float32)
  CONST031 = tl.full([], -8.49632273398321, dtype=tl.float32)
  CONST032 = tl.full([], -5.20291384706685, dtype=tl.float32)
  CONST033 = tl.full([], -3.46860923137790, dtype=tl.float32)
  CONST034 = tl.full([], -1.73430461568895, dtype=tl.float32)
  VAR05 = x * x * x * x * x
  VAR06 = x * x * x * x
  VAR07 = x * x * x
  VAR08 = x * x
  VAR14 = y * y * y * y * y
  VAR15 = y * y * y * y
  VAR16 = y * y * y
  VAR17 = y * y
  VAR23 = z * z * z * z * z
  VAR24 = z * z * z * z
  VAR25 = z * z * z
  VAR26 = z * z
  Y00 = CONST001 * VAR05 + CONST009 * VAR24 * x + CONST023 * VAR07 * VAR26
  Y01 = y * (CONST022 * VAR07 * z - CONST022 * VAR25 * x)
  Y02 = CONST000 * VAR05 +
    VAR07 * (CONST028 * VAR17 + CONST033 * VAR26) +
    x * (-CONST021 * VAR17 * VAR26 + CONST032 * VAR24)
  Y03 = CONST027 * VAR07 * y * z +
    x * (CONST017 * VAR16 * z + CONST026 * VAR25 * y)
  Y04 = CONST002 * VAR05 +
    VAR07 * (CONST003 * VAR26 + CONST025 * VAR17) +
    x * (CONST002 * VAR24 + CONST010 * VAR15 + CONST024 * VAR17 * VAR26)
  Y05 = CONST004 * VAR14 +
    VAR16 * (CONST029 * VAR08 + CONST029 * VAR26) +
    y * (CONST005 * VAR06 + CONST006 * VAR24 + CONST011 * VAR08 * VAR26)
  Y06 = CONST002 * VAR23 +
    VAR25 * (CONST003 * VAR08 + CONST024 * VAR17) +
    z * (CONST007 * VAR06 + CONST012 * VAR15 + CONST024 * VAR08 * VAR17)
  Y07 = VAR16 * (CONST026 * VAR08 - CONST026 * VAR26) + y * (-CONST031 * VAR06 + CONST031 * VAR24)
  Y08 = CONST034 * VAR23 +
    VAR25 * (CONST013 * VAR17 + CONST030 * VAR08) +
    z * (CONST021 * VAR08 * VAR17 - CONST032 * VAR06)
  Y09 = y * (CONST018 * VAR06 + CONST018 * VAR24 + CONST020 * VAR08 * VAR26)
  Y10 = CONST001 * VAR23 + CONST009 * VAR06 * z + CONST023 * VAR08 * VAR25
  output_striding = tl.arange(0, $(block_size)) * $(output_stride)
  output_row_offset = output_striding + $(block_size) * $(output_stride) * block_id +
    $(col_offset)
  tl.store(output_ptr + output_row_offset, Y00, mask=output_row_offset < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(1), Y01,
    mask=output_row_offset + $(1) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(2), Y02,
    mask=output_row_offset + $(2) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(3), Y03,
    mask=output_row_offset + $(3) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(4), Y04,
    mask=output_row_offset + $(4) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(5), Y05,
    mask=output_row_offset + $(5) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(6), Y06,
    mask=output_row_offset + $(6) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(7), Y07,
    mask=output_row_offset + $(7) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(8), Y08,
    mask=output_row_offset + $(8) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(9), Y09,
    mask=output_row_offset + $(9) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(10), Y10,
    mask=output_row_offset + $(10) < $(output_numel))
}
def kernel (p : Parameters) := source p.input p.output p.blockSize p.coordNumel p.outputNumel p.columnOffset p.outputStride
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
