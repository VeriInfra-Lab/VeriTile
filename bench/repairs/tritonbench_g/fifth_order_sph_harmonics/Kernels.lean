import bench.tritonbench_g.fifth_order_sph_harmonics.Kernels

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 48000
namespace Original
abbrev Parameters := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.Parameters
abbrev source := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.source
abbrev kernel := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.kernel
end Original

/-- The complete unchanged eleven-channel forward arithmetic is reused.
Actual int64 pid/lane widening has its natural-index projection here; Native
checks every fixed-width product, sum and active byte address separately. -/
abbrev forwardSource := Original.source

/-- Complete backward source: original operation order, 29 corrected decimal
coefficient occurrences, three old-zero gradient loads and three stores.
The Python launch explicitly selects enable_fp_fusion=False in both phases. -/
def backwardSource (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride upstream_row_stride upstream_col_stride : Nat) : ComputeKernel := triton {
  tl.region coord_ptr = tl.float32, coord_grad_ptr = tl.float32, sph_grad_ptr = tl.float32
  block_id = tl.program_id(0)
  coord_stride = $(3)
  coord_striding = (tl.arange(0, $(block_size)) * coord_stride)
  coord_row_offset = (coord_striding + (($(block_size) * coord_stride) * block_id))
  x = tl.load((coord_ptr + coord_row_offset), mask=coord_row_offset < $(coord_numel))
  y = tl.load(((coord_ptr + coord_row_offset) + $(1)), mask=(coord_row_offset + $(1)) < $(coord_numel))
  z = tl.load(((coord_ptr + coord_row_offset) + $(2)), mask=(coord_row_offset + $(2)) < $(coord_numel))
  output_striding = (tl.arange(0, $(block_size)) * $(output_stride))
  output_row_offset = ((output_striding + (($(block_size) * $(output_stride)) * block_id)) + $(col_offset))
  upstream_row_offset = (((tl.arange(0, $(block_size)) + ($(block_size) * block_id)) * $(upstream_row_stride)) + ($(col_offset) * $(upstream_col_stride)))
  g_0 = tl.load((sph_grad_ptr + upstream_row_offset), mask=output_row_offset < $(output_numel))
  g_1 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(1) * $(upstream_col_stride))), mask=(output_row_offset + $(1)) < $(output_numel))
  g_2 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(2) * $(upstream_col_stride))), mask=(output_row_offset + $(2)) < $(output_numel))
  g_3 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(3) * $(upstream_col_stride))), mask=(output_row_offset + $(3)) < $(output_numel))
  g_4 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(4) * $(upstream_col_stride))), mask=(output_row_offset + $(4)) < $(output_numel))
  g_5 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(5) * $(upstream_col_stride))), mask=(output_row_offset + $(5)) < $(output_numel))
  g_6 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(6) * $(upstream_col_stride))), mask=(output_row_offset + $(6)) < $(output_numel))
  g_7 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(7) * $(upstream_col_stride))), mask=(output_row_offset + $(7)) < $(output_numel))
  g_8 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(8) * $(upstream_col_stride))), mask=(output_row_offset + $(8)) < $(output_numel))
  g_9 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(9) * $(upstream_col_stride))), mask=(output_row_offset + $(9)) < $(output_numel))
  g_10 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(10) * $(upstream_col_stride))), mask=(output_row_offset + $(10)) < $(output_numel))
  CONST000 = tl.full([], 1.60565407233314, dtype=tl.float32)
  CONST001 = tl.full([], 3.00000000000000, dtype=tl.float32)
  CONST002 = tl.full([], 3.21130814466628, dtype=tl.float32)
  CONST003 = tl.full([], 1.60565407233314, dtype=tl.float32)
  CONST004 = tl.full([], 6.42261628933256, dtype=tl.float32)
  CONST005 = tl.full([], 6.42261628933256, dtype=tl.float32)
  CONST006 = tl.full([], 8.67152307844476, dtype=tl.float32)
  CONST007 = tl.full([], 8.02827036166571, dtype=tl.float32)
  CONST008 = tl.full([], 6.93721846275580, dtype=tl.float32)
  CONST009 = tl.full([], 11.6340690431164, dtype=tl.float32)
  CONST010 = tl.full([], 12.8452325786651, dtype=tl.float32)
  CONST011 = tl.full([], 6.21867148191637, dtype=tl.float32)
  CONST012 = tl.full([], 6.21867148191637, dtype=tl.float32)
  CONST014 = tl.full([], 12.4373429638327, dtype=tl.float32)
  CONST017 = tl.full([], 12.8452325786651, dtype=tl.float32)
  CONST018 = tl.full([], 13.8744369255116, dtype=tl.float32)
  CONST019 = tl.full([], 24.8746859276655, dtype=tl.float32)
  CONST020 = tl.full([], 24.8746859276655, dtype=tl.float32)
  CONST021 = tl.full([], 27.7488738510232, dtype=tl.float32)
  CONST024 = tl.full([], 29.4321253055229, dtype=tl.float32)
  CONST027 = tl.full([], 7.35803132638072, dtype=tl.float32)
  CONST029 = tl.full([], 46.5362761724657, dtype=tl.float32)
  CONST030 = tl.full([], 51.3809303146605, dtype=tl.float32)
  CONST031 = tl.full([], 51.3809303146605, dtype=tl.float32)
  CONST034 = tl.full([], 101.955872807799, dtype=tl.float32)
  CONST036 = tl.full([], -8.67152307844475, dtype=tl.float32)
  CONST037 = tl.full([], 3.46860923137790, dtype=tl.float32)
  CONST038 = tl.full([], -88.2963759165686, dtype=tl.float32)
  CONST039 = tl.full([], -83.2466215530696, dtype=tl.float32)
  CONST040 = tl.full([], -69.8044142586986, dtype=tl.float32)
  CONST041 = tl.full([], -50.9779364038993, dtype=tl.float32)
  CONST042 = tl.full([], -50.9779364038993, dtype=tl.float32)
  CONST043 = tl.full([], -46.5362761724657, dtype=tl.float32)
  CONST044 = tl.full([], -44.1481879582843, dtype=tl.float32)
  CONST045 = tl.full([], -41.6233107765348, dtype=tl.float32)
  CONST046 = tl.full([], -38.5356977359954, dtype=tl.float32)
  CONST047 = tl.full([], -38.5356977359954, dtype=tl.float32)
  CONST048 = tl.full([], -33.1662479035540, dtype=tl.float32)
  CONST049 = tl.full([], -33.9852909359329, dtype=tl.float32)
  CONST050 = tl.full([], 6.42261628933257, dtype=tl.float32)
  CONST051 = tl.full([], -33.9852909359329, dtype=tl.float32)
  CONST052 = tl.full([], -29.4321253055229, dtype=tl.float32)
  CONST053 = tl.full([], -27.7488738510232, dtype=tl.float32)
  CONST054 = tl.full([], -20.8116553882674, dtype=tl.float32)
  CONST055 = tl.full([], -19.2678488679977, dtype=tl.float32)
  CONST056 = tl.full([], -19.2678488679977, dtype=tl.float32)
  CONST057 = tl.full([], -16.9926454679664, dtype=tl.float32)
  CONST058 = tl.full([], -16.9926454679664, dtype=tl.float32)
  CONST059 = tl.full([], -13.8744369255116, dtype=tl.float32)
  CONST060 = tl.full([], -16.5831239517770, dtype=tl.float32)
  CONST061 = tl.full([], -8.49632273398321, dtype=tl.float32)
  CONST062 = tl.full([], -6.93721846275580, dtype=tl.float32)
  CONST063 = tl.full([], -5.20291384706685, dtype=tl.float32)
  CONST064 = tl.full([], -3.46860923137790, dtype=tl.float32)
  EXACT000 = tl.full([], 11.63406904311645, dtype=tl.float32)
  EXACT001 = tl.full([], -69.8044142586987, dtype=tl.float32)
  EXACT002 = tl.full([], -88.2963759165687, dtype=tl.float32)
  EXACT003 = tl.full([], 46.5362761724656, dtype=tl.float32)
  EXACT004 = tl.full([], -46.5362761724658, dtype=tl.float32)
  EXACT005 = tl.full([], 8.67152307844475, dtype=tl.float32)
  EXACT006 = tl.full([], -50.9779364038992, dtype=tl.float32)
  EXACT007 = tl.full([], 8.0282703616657, dtype=tl.float32)
  EXACT008 = tl.full([], 24.87468592766548, dtype=tl.float32)
  EXACT009 = tl.full([], 24.8746859276654, dtype=tl.float32)
  EXACT010 = tl.full([], -33.9852909359328, dtype=tl.float32)
  EXACT011 = tl.full([], -33.98529093593284, dtype=tl.float32)
  EXACT012 = tl.full([], 29.43212530552288, dtype=tl.float32)
  EXACT013 = tl.full([], 101.9558728077987, dtype=tl.float32)
  EXACT014 = tl.full([], 51.3809303146604, dtype=tl.float32)
  EXACT015 = tl.full([], 51.3809303146604, dtype=tl.float32)
  EXACT016 = tl.full([], 46.5362761724656, dtype=tl.float32)
  EXACT017 = tl.full([], -46.5362761724658, dtype=tl.float32)
  EXACT018 = tl.full([], -88.2963759165687, dtype=tl.float32)
  EXACT019 = tl.full([], 11.63406904311645, dtype=tl.float32)
  EXACT020 = tl.full([], -69.8044142586987, dtype=tl.float32)
  EXACT021 = tl.full([], -50.9779364038992, dtype=tl.float32)
  EXACT022 = tl.full([], 6.42261628933256, dtype=tl.float32)
  EXACT023 = tl.full([], 24.8746859276654, dtype=tl.float32)
  EXACT024 = tl.full([], 24.87468592766548, dtype=tl.float32)
  EXACT025 = tl.full([], 8.0282703616657, dtype=tl.float32)
  EXACT026 = tl.full([], -33.9852909359328, dtype=tl.float32)
  EXACT027 = tl.full([], -33.98529093593284, dtype=tl.float32)
  EXACT028 = tl.full([], 29.43212530552288, dtype=tl.float32)
  VAR06 = (((x * x) * x) * x)
  VAR07 = ((x * x) * x)
  VAR08 = (x * x)
  VAR15 = (((y * y) * y) * y)
  VAR16 = ((y * y) * y)
  VAR17 = (y * y)
  VAR24 = (((z * z) * z) * z)
  VAR25 = ((z * z) * z)
  VAR26 = (z * z)
  g_x = tl.load((coord_grad_ptr + coord_row_offset), mask=coord_row_offset < $(coord_numel))
  g_y = tl.load(((coord_grad_ptr + coord_row_offset) + $(1)), mask=(coord_row_offset + $(1)) < $(coord_numel))
  g_z = tl.load(((coord_grad_ptr + coord_row_offset) + $(2)), mask=(coord_row_offset + $(2)) < $(coord_numel))
  g_x = g_x + (((((((((((g_0 * (((EXACT000 * VAR06) + (CONST009 * VAR24)) + ((EXACT001 * VAR08) * VAR26))) + ((g_1 * y) * (((EXACT002 * VAR08) * z) - (CONST052 * VAR25)))) + (g_10 * (((EXACT003 * VAR07) * z) + ((EXACT004 * VAR25) * x)))) + (g_2 * (((((CONST001 * VAR08) * ((CONST059 * VAR17) + (CONST064 * VAR26))) + (EXACT005 * VAR06)) - ((CONST045 * VAR17) * VAR26)) + (CONST063 * VAR24)))) + (g_3 * (((((EXACT006 * VAR08) * y) * z) - ((CONST049 * VAR16) * z)) + ((CONST057 * VAR25) * y)))) + (g_4 * (((((CONST000 * VAR24) + ((CONST001 * VAR08) * ((CONST002 * VAR26) + (CONST055 * VAR17)))) + (EXACT007 * VAR06)) + (CONST010 * VAR15)) + ((CONST056 * VAR17) * VAR26)))) + (g_5 * (((CONST048 * VAR16) * x) + (y * ((EXACT008 * VAR07) + ((EXACT009 * VAR26) * x)))))) + (g_6 * (((CONST005 * VAR25) * x) + (z * ((CONST004 * VAR07) + ((CONST046 * VAR17) * x)))))) + (g_7 * (((EXACT010 * VAR16) * x) - ((EXACT011 * VAR07) * y)))) + (g_8 * (((CONST008 * VAR25) * x) + (z * (((CONST039 * VAR17) * x) - (CONST054 * VAR07)))))) + ((g_9 * y) * ((EXACT012 * VAR07) + ((CONST038 * VAR26) * x))))
  g_y = g_y + (((((((((g_1 * (((CONST052 * VAR07) * z) - ((CONST052 * VAR25) * x))) + (g_2 * (((((-CONST039) * VAR26) * x) * y) + ((CONST053 * VAR07) * y)))) + (g_3 * (((CONST058 * VAR07) * z) + (x * (((EXACT013 * VAR17) * z) + (CONST057 * VAR25)))))) + (g_4 * (((CONST047 * VAR07) * y) + (x * ((EXACT014 * VAR16) + ((CONST046 * VAR26) * y)))))) + (g_5 * ((((((CONST001 * VAR17) * ((CONST060 * VAR08) + (CONST060 * VAR26))) + (CONST011 * VAR06)) + (CONST012 * VAR24)) + ((CONST014 * VAR08) * VAR26)) - (CONST060 * VAR15)))) + (g_6 * (((CONST046 * VAR25) * y) + (z * ((EXACT015 * VAR16) + ((CONST046 * VAR08) * y)))))) + (g_7 * ((((CONST001 * VAR17) * ((CONST057 * VAR08) - (CONST057 * VAR26))) - (CONST061 * VAR06)) + (CONST061 * VAR24)))) + (g_8 * (((CONST021 * VAR25) * y) + (((CONST039 * VAR08) * y) * z)))) + (g_9 * (((CONST027 * VAR06) + (CONST027 * VAR24)) + ((CONST044 * VAR08) * VAR26))))
  g_z = g_z + (((((((((((g_0 * (((EXACT016 * VAR25) * x) + ((EXACT017 * VAR07) * z))) + ((g_1 * y) * ((((-EXACT018) * VAR26) * x) + (CONST052 * VAR07)))) + (g_10 * (((CONST009 * VAR06) + (EXACT019 * VAR24)) + ((EXACT020 * VAR08) * VAR26)))) + (g_2 * (((CONST062 * VAR07) * z) + (x * ((((-CONST039) * VAR17) * z) + (CONST054 * VAR25)))))) + (g_3 * (((CONST058 * VAR07) * y) + (x * (((EXACT021 * VAR26) * y) - (CONST049 * VAR16)))))) + (g_4 * (((CONST005 * VAR07) * z) + (x * (((CONST046 * VAR17) * z) + (EXACT022 * VAR25)))))) + (g_5 * (((CONST048 * VAR16) * z) + (y * (((EXACT023 * VAR08) * z) + (EXACT024 * VAR25)))))) + (g_6 * ((((((CONST001 * VAR26) * ((CONST002 * VAR08) + (CONST056 * VAR17))) + (CONST003 * VAR06)) + (EXACT025 * VAR24)) + (CONST017 * VAR15)) + ((CONST056 * VAR08) * VAR17)))) + (g_7 * ((((-EXACT026) * VAR16) * z) + ((EXACT027 * VAR25) * y)))) + (g_8 * (((((CONST001 * VAR26) * ((CONST018 * VAR17) + (CONST037 * VAR08))) + (CONST036 * VAR24)) + ((CONST045 * VAR08) * VAR17)) - (CONST063 * VAR06)))) + ((g_9 * y) * ((EXACT028 * VAR25) + ((CONST038 * VAR08) * z))))
  tl.store((coord_grad_ptr + coord_row_offset), g_x, mask=coord_row_offset < $(coord_numel))
  tl.store(((coord_grad_ptr + coord_row_offset) + $(1)), g_y, mask=(coord_row_offset + $(1)) < $(coord_numel))
  tl.store(((coord_grad_ptr + coord_row_offset) + $(2)), g_z, mask=(coord_row_offset + $(2)) < $(coord_numel))
}

/-- Share left-associated rounded powers. The numerical operation tree of every
used power is unchanged; recomputation is removed without reassociation. -/
def forwardPowersSource (coord_ptr output_ptr : RegionName)
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
  VAR08 = x * x
  VAR07 = VAR08 * x
  VAR06 = VAR07 * x
  VAR05 = VAR06 * x
  VAR17 = y * y
  VAR16 = VAR17 * y
  VAR15 = VAR16 * y
  VAR14 = VAR15 * y
  VAR26 = z * z
  VAR25 = VAR26 * z
  VAR24 = VAR25 * z
  VAR23 = VAR24 * z
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
def backwardPowersSource (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride upstream_row_stride upstream_col_stride : Nat) : ComputeKernel := triton {
  tl.region coord_ptr = tl.float32, coord_grad_ptr = tl.float32, sph_grad_ptr = tl.float32
  block_id = tl.program_id(0)
  coord_stride = $(3)
  coord_striding = (tl.arange(0, $(block_size)) * coord_stride)
  coord_row_offset = (coord_striding + (($(block_size) * coord_stride) * block_id))
  x = tl.load((coord_ptr + coord_row_offset), mask=coord_row_offset < $(coord_numel))
  y = tl.load(((coord_ptr + coord_row_offset) + $(1)), mask=(coord_row_offset + $(1)) < $(coord_numel))
  z = tl.load(((coord_ptr + coord_row_offset) + $(2)), mask=(coord_row_offset + $(2)) < $(coord_numel))
  output_striding = (tl.arange(0, $(block_size)) * $(output_stride))
  output_row_offset = ((output_striding + (($(block_size) * $(output_stride)) * block_id)) + $(col_offset))
  upstream_row_offset = (((tl.arange(0, $(block_size)) + ($(block_size) * block_id)) * $(upstream_row_stride)) + ($(col_offset) * $(upstream_col_stride)))
  g_0 = tl.load((sph_grad_ptr + upstream_row_offset), mask=output_row_offset < $(output_numel))
  g_1 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(1) * $(upstream_col_stride))), mask=(output_row_offset + $(1)) < $(output_numel))
  g_2 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(2) * $(upstream_col_stride))), mask=(output_row_offset + $(2)) < $(output_numel))
  g_3 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(3) * $(upstream_col_stride))), mask=(output_row_offset + $(3)) < $(output_numel))
  g_4 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(4) * $(upstream_col_stride))), mask=(output_row_offset + $(4)) < $(output_numel))
  g_5 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(5) * $(upstream_col_stride))), mask=(output_row_offset + $(5)) < $(output_numel))
  g_6 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(6) * $(upstream_col_stride))), mask=(output_row_offset + $(6)) < $(output_numel))
  g_7 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(7) * $(upstream_col_stride))), mask=(output_row_offset + $(7)) < $(output_numel))
  g_8 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(8) * $(upstream_col_stride))), mask=(output_row_offset + $(8)) < $(output_numel))
  g_9 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(9) * $(upstream_col_stride))), mask=(output_row_offset + $(9)) < $(output_numel))
  g_10 = tl.load(((sph_grad_ptr + upstream_row_offset) + ($(10) * $(upstream_col_stride))), mask=(output_row_offset + $(10)) < $(output_numel))
  CONST000 = tl.full([], 1.60565407233314, dtype=tl.float32)
  CONST001 = tl.full([], 3.00000000000000, dtype=tl.float32)
  CONST002 = tl.full([], 3.21130814466628, dtype=tl.float32)
  CONST003 = tl.full([], 1.60565407233314, dtype=tl.float32)
  CONST004 = tl.full([], 6.42261628933256, dtype=tl.float32)
  CONST005 = tl.full([], 6.42261628933256, dtype=tl.float32)
  CONST006 = tl.full([], 8.67152307844476, dtype=tl.float32)
  CONST007 = tl.full([], 8.02827036166571, dtype=tl.float32)
  CONST008 = tl.full([], 6.93721846275580, dtype=tl.float32)
  CONST009 = tl.full([], 11.6340690431164, dtype=tl.float32)
  CONST010 = tl.full([], 12.8452325786651, dtype=tl.float32)
  CONST011 = tl.full([], 6.21867148191637, dtype=tl.float32)
  CONST012 = tl.full([], 6.21867148191637, dtype=tl.float32)
  CONST014 = tl.full([], 12.4373429638327, dtype=tl.float32)
  CONST017 = tl.full([], 12.8452325786651, dtype=tl.float32)
  CONST018 = tl.full([], 13.8744369255116, dtype=tl.float32)
  CONST019 = tl.full([], 24.8746859276655, dtype=tl.float32)
  CONST020 = tl.full([], 24.8746859276655, dtype=tl.float32)
  CONST021 = tl.full([], 27.7488738510232, dtype=tl.float32)
  CONST024 = tl.full([], 29.4321253055229, dtype=tl.float32)
  CONST027 = tl.full([], 7.35803132638072, dtype=tl.float32)
  CONST029 = tl.full([], 46.5362761724657, dtype=tl.float32)
  CONST030 = tl.full([], 51.3809303146605, dtype=tl.float32)
  CONST031 = tl.full([], 51.3809303146605, dtype=tl.float32)
  CONST034 = tl.full([], 101.955872807799, dtype=tl.float32)
  CONST036 = tl.full([], -8.67152307844475, dtype=tl.float32)
  CONST037 = tl.full([], 3.46860923137790, dtype=tl.float32)
  CONST038 = tl.full([], -88.2963759165686, dtype=tl.float32)
  CONST039 = tl.full([], -83.2466215530696, dtype=tl.float32)
  CONST040 = tl.full([], -69.8044142586986, dtype=tl.float32)
  CONST041 = tl.full([], -50.9779364038993, dtype=tl.float32)
  CONST042 = tl.full([], -50.9779364038993, dtype=tl.float32)
  CONST043 = tl.full([], -46.5362761724657, dtype=tl.float32)
  CONST044 = tl.full([], -44.1481879582843, dtype=tl.float32)
  CONST045 = tl.full([], -41.6233107765348, dtype=tl.float32)
  CONST046 = tl.full([], -38.5356977359954, dtype=tl.float32)
  CONST047 = tl.full([], -38.5356977359954, dtype=tl.float32)
  CONST048 = tl.full([], -33.1662479035540, dtype=tl.float32)
  CONST049 = tl.full([], -33.9852909359329, dtype=tl.float32)
  CONST050 = tl.full([], 6.42261628933257, dtype=tl.float32)
  CONST051 = tl.full([], -33.9852909359329, dtype=tl.float32)
  CONST052 = tl.full([], -29.4321253055229, dtype=tl.float32)
  CONST053 = tl.full([], -27.7488738510232, dtype=tl.float32)
  CONST054 = tl.full([], -20.8116553882674, dtype=tl.float32)
  CONST055 = tl.full([], -19.2678488679977, dtype=tl.float32)
  CONST056 = tl.full([], -19.2678488679977, dtype=tl.float32)
  CONST057 = tl.full([], -16.9926454679664, dtype=tl.float32)
  CONST058 = tl.full([], -16.9926454679664, dtype=tl.float32)
  CONST059 = tl.full([], -13.8744369255116, dtype=tl.float32)
  CONST060 = tl.full([], -16.5831239517770, dtype=tl.float32)
  CONST061 = tl.full([], -8.49632273398321, dtype=tl.float32)
  CONST062 = tl.full([], -6.93721846275580, dtype=tl.float32)
  CONST063 = tl.full([], -5.20291384706685, dtype=tl.float32)
  CONST064 = tl.full([], -3.46860923137790, dtype=tl.float32)
  EXACT000 = tl.full([], 11.63406904311645, dtype=tl.float32)
  EXACT001 = tl.full([], -69.8044142586987, dtype=tl.float32)
  EXACT002 = tl.full([], -88.2963759165687, dtype=tl.float32)
  EXACT003 = tl.full([], 46.5362761724656, dtype=tl.float32)
  EXACT004 = tl.full([], -46.5362761724658, dtype=tl.float32)
  EXACT005 = tl.full([], 8.67152307844475, dtype=tl.float32)
  EXACT006 = tl.full([], -50.9779364038992, dtype=tl.float32)
  EXACT007 = tl.full([], 8.0282703616657, dtype=tl.float32)
  EXACT008 = tl.full([], 24.87468592766548, dtype=tl.float32)
  EXACT009 = tl.full([], 24.8746859276654, dtype=tl.float32)
  EXACT010 = tl.full([], -33.9852909359328, dtype=tl.float32)
  EXACT011 = tl.full([], -33.98529093593284, dtype=tl.float32)
  EXACT012 = tl.full([], 29.43212530552288, dtype=tl.float32)
  EXACT013 = tl.full([], 101.9558728077987, dtype=tl.float32)
  EXACT014 = tl.full([], 51.3809303146604, dtype=tl.float32)
  EXACT015 = tl.full([], 51.3809303146604, dtype=tl.float32)
  EXACT016 = tl.full([], 46.5362761724656, dtype=tl.float32)
  EXACT017 = tl.full([], -46.5362761724658, dtype=tl.float32)
  EXACT018 = tl.full([], -88.2963759165687, dtype=tl.float32)
  EXACT019 = tl.full([], 11.63406904311645, dtype=tl.float32)
  EXACT020 = tl.full([], -69.8044142586987, dtype=tl.float32)
  EXACT021 = tl.full([], -50.9779364038992, dtype=tl.float32)
  EXACT022 = tl.full([], 6.42261628933256, dtype=tl.float32)
  EXACT023 = tl.full([], 24.8746859276654, dtype=tl.float32)
  EXACT024 = tl.full([], 24.87468592766548, dtype=tl.float32)
  EXACT025 = tl.full([], 8.0282703616657, dtype=tl.float32)
  EXACT026 = tl.full([], -33.9852909359328, dtype=tl.float32)
  EXACT027 = tl.full([], -33.98529093593284, dtype=tl.float32)
  EXACT028 = tl.full([], 29.43212530552288, dtype=tl.float32)
  VAR08 = x * x
  VAR07 = VAR08 * x
  VAR06 = VAR07 * x
  VAR17 = y * y
  VAR16 = VAR17 * y
  VAR15 = VAR16 * y
  VAR26 = z * z
  VAR25 = VAR26 * z
  VAR24 = VAR25 * z
  g_x = tl.load((coord_grad_ptr + coord_row_offset), mask=coord_row_offset < $(coord_numel))
  g_y = tl.load(((coord_grad_ptr + coord_row_offset) + $(1)), mask=(coord_row_offset + $(1)) < $(coord_numel))
  g_z = tl.load(((coord_grad_ptr + coord_row_offset) + $(2)), mask=(coord_row_offset + $(2)) < $(coord_numel))
  g_x = g_x + (((((((((((g_0 * (((EXACT000 * VAR06) + (CONST009 * VAR24)) + ((EXACT001 * VAR08) * VAR26))) + ((g_1 * y) * (((EXACT002 * VAR08) * z) - (CONST052 * VAR25)))) + (g_10 * (((EXACT003 * VAR07) * z) + ((EXACT004 * VAR25) * x)))) + (g_2 * (((((CONST001 * VAR08) * ((CONST059 * VAR17) + (CONST064 * VAR26))) + (EXACT005 * VAR06)) - ((CONST045 * VAR17) * VAR26)) + (CONST063 * VAR24)))) + (g_3 * (((((EXACT006 * VAR08) * y) * z) - ((CONST049 * VAR16) * z)) + ((CONST057 * VAR25) * y)))) + (g_4 * (((((CONST000 * VAR24) + ((CONST001 * VAR08) * ((CONST002 * VAR26) + (CONST055 * VAR17)))) + (EXACT007 * VAR06)) + (CONST010 * VAR15)) + ((CONST056 * VAR17) * VAR26)))) + (g_5 * (((CONST048 * VAR16) * x) + (y * ((EXACT008 * VAR07) + ((EXACT009 * VAR26) * x)))))) + (g_6 * (((CONST005 * VAR25) * x) + (z * ((CONST004 * VAR07) + ((CONST046 * VAR17) * x)))))) + (g_7 * (((EXACT010 * VAR16) * x) - ((EXACT011 * VAR07) * y)))) + (g_8 * (((CONST008 * VAR25) * x) + (z * (((CONST039 * VAR17) * x) - (CONST054 * VAR07)))))) + ((g_9 * y) * ((EXACT012 * VAR07) + ((CONST038 * VAR26) * x))))
  g_y = g_y + (((((((((g_1 * (((CONST052 * VAR07) * z) - ((CONST052 * VAR25) * x))) + (g_2 * (((((-CONST039) * VAR26) * x) * y) + ((CONST053 * VAR07) * y)))) + (g_3 * (((CONST058 * VAR07) * z) + (x * (((EXACT013 * VAR17) * z) + (CONST057 * VAR25)))))) + (g_4 * (((CONST047 * VAR07) * y) + (x * ((EXACT014 * VAR16) + ((CONST046 * VAR26) * y)))))) + (g_5 * ((((((CONST001 * VAR17) * ((CONST060 * VAR08) + (CONST060 * VAR26))) + (CONST011 * VAR06)) + (CONST012 * VAR24)) + ((CONST014 * VAR08) * VAR26)) - (CONST060 * VAR15)))) + (g_6 * (((CONST046 * VAR25) * y) + (z * ((EXACT015 * VAR16) + ((CONST046 * VAR08) * y)))))) + (g_7 * ((((CONST001 * VAR17) * ((CONST057 * VAR08) - (CONST057 * VAR26))) - (CONST061 * VAR06)) + (CONST061 * VAR24)))) + (g_8 * (((CONST021 * VAR25) * y) + (((CONST039 * VAR08) * y) * z)))) + (g_9 * (((CONST027 * VAR06) + (CONST027 * VAR24)) + ((CONST044 * VAR08) * VAR26))))
  g_z = g_z + (((((((((((g_0 * (((EXACT016 * VAR25) * x) + ((EXACT017 * VAR07) * z))) + ((g_1 * y) * ((((-EXACT018) * VAR26) * x) + (CONST052 * VAR07)))) + (g_10 * (((CONST009 * VAR06) + (EXACT019 * VAR24)) + ((EXACT020 * VAR08) * VAR26)))) + (g_2 * (((CONST062 * VAR07) * z) + (x * ((((-CONST039) * VAR17) * z) + (CONST054 * VAR25)))))) + (g_3 * (((CONST058 * VAR07) * y) + (x * (((EXACT021 * VAR26) * y) - (CONST049 * VAR16)))))) + (g_4 * (((CONST005 * VAR07) * z) + (x * (((CONST046 * VAR17) * z) + (EXACT022 * VAR25)))))) + (g_5 * (((CONST048 * VAR16) * z) + (y * (((EXACT023 * VAR08) * z) + (EXACT024 * VAR25)))))) + (g_6 * ((((((CONST001 * VAR26) * ((CONST002 * VAR08) + (CONST056 * VAR17))) + (CONST003 * VAR06)) + (EXACT025 * VAR24)) + (CONST017 * VAR15)) + ((CONST056 * VAR08) * VAR17)))) + (g_7 * ((((-EXACT026) * VAR16) * z) + ((EXACT027 * VAR25) * y)))) + (g_8 * (((((CONST001 * VAR26) * ((CONST018 * VAR17) + (CONST037 * VAR08))) + (CONST036 * VAR24)) + ((CONST045 * VAR08) * VAR17)) - (CONST063 * VAR06)))) + ((g_9 * y) * ((EXACT028 * VAR25) + ((CONST038 * VAR08) * z))))
  tl.store((coord_grad_ptr + coord_row_offset), g_x, mask=coord_row_offset < $(coord_numel))
  tl.store(((coord_grad_ptr + coord_row_offset) + $(1)), g_y, mask=(coord_row_offset + $(1)) < $(coord_numel))
  tl.store(((coord_grad_ptr + coord_row_offset) + $(2)), g_z, mask=(coord_row_offset + $(2)) < $(coord_numel))
}

def coordinateRegion : RegionName := "coordinates"
def outputRegion : RegionName := "output"
def upstreamRegion : RegionName := "upstream"
def gradientRegion : RegionName := "gradient"

structure Parameters where
  rows : Nat
  blockSize : Nat
  blockPositive : 0 < blockSize
  outputWidth : Nat
  columnOffset : Nat
  upstreamRowStride : Nat
  upstreamColStride : Nat

def forwardParameters (p : Parameters) : Original.Parameters :=
  ⟨coordinateRegion,outputRegion,p.blockSize,3*p.rows,p.outputWidth*p.rows,p.columnOffset,p.outputWidth⟩

def forwardKernel (p : Parameters) := Original.kernel (forwardParameters p)

def backwardKernel (p : Parameters) :=
  backwardSource coordinateRegion gradientRegion upstreamRegion
    p.blockSize (3*p.rows) (p.outputWidth*p.rows) p.columnOffset p.outputWidth
    p.upstreamRowStride p.upstreamColStride

def forwardPowersKernel (p : Parameters) :=
  forwardPowersSource coordinateRegion outputRegion p.blockSize (3*p.rows)
    (p.outputWidth*p.rows) p.columnOffset p.outputWidth

def backwardPowersKernel (p : Parameters) :=
  backwardPowersSource coordinateRegion gradientRegion upstreamRegion
    p.blockSize (3*p.rows) (p.outputWidth*p.rows) p.columnOffset p.outputWidth
    p.upstreamRowStride p.upstreamColStride

end VeriTile.Bench.Repairs.FifthOrderSphHarmonics
