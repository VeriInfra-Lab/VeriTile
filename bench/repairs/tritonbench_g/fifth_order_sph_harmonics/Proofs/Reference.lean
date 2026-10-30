import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Kernels
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedReference

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Reference
open VeriTile.Triton FP.Structural
noncomputable section

abbrev forwardValues {α : Type} (M : Algebra α) (x y z : α) : Fin 11 → α :=
  VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.channels M x y z

/-- Source-order backward arithmetic on already-loaded words. This is separate
from the independent polynomial/Jacobian target in Real. -/
def backwardValues {α : Type} (M : Algebra α) (x y z : α)
    (g : Fin 11 → α) (initial : Fin 3 → α) : Fin 3 → α :=
  let add := M.binary (some .fp32) .real .add
  let sub := M.binary (some .fp32) .real .sub
  let mul := M.binary (some .fp32) .real .mul
  let neg := M.unary (some .fp32) (.neg .real)
  let g_0 := g 0
  let g_1 := g 1
  let g_2 := g 2
  let g_3 := g 3
  let g_4 := g 4
  let g_5 := g 5
  let g_6 := g 6
  let g_7 := g 7
  let g_8 := g 8
  let g_9 := g 9
  let g_10 := g 10
  let CONST000 := M.literal (some .fp32) .real (1.60565407233314)
  let CONST001 := M.literal (some .fp32) .real (3.00000000000000)
  let CONST002 := M.literal (some .fp32) .real (3.21130814466628)
  let CONST003 := M.literal (some .fp32) .real (1.60565407233314)
  let CONST004 := M.literal (some .fp32) .real (6.42261628933256)
  let CONST005 := M.literal (some .fp32) .real (6.42261628933256)
  let CONST006 := M.literal (some .fp32) .real (8.67152307844476)
  let CONST007 := M.literal (some .fp32) .real (8.02827036166571)
  let CONST008 := M.literal (some .fp32) .real (6.93721846275580)
  let CONST009 := M.literal (some .fp32) .real (11.6340690431164)
  let CONST010 := M.literal (some .fp32) .real (12.8452325786651)
  let CONST011 := M.literal (some .fp32) .real (6.21867148191637)
  let CONST012 := M.literal (some .fp32) .real (6.21867148191637)
  let CONST014 := M.literal (some .fp32) .real (12.4373429638327)
  let CONST017 := M.literal (some .fp32) .real (12.8452325786651)
  let CONST018 := M.literal (some .fp32) .real (13.8744369255116)
  let CONST019 := M.literal (some .fp32) .real (24.8746859276655)
  let CONST020 := M.literal (some .fp32) .real (24.8746859276655)
  let CONST021 := M.literal (some .fp32) .real (27.7488738510232)
  let CONST024 := M.literal (some .fp32) .real (29.4321253055229)
  let CONST027 := M.literal (some .fp32) .real (7.35803132638072)
  let CONST029 := M.literal (some .fp32) .real (46.5362761724657)
  let CONST030 := M.literal (some .fp32) .real (51.3809303146605)
  let CONST031 := M.literal (some .fp32) .real (51.3809303146605)
  let CONST034 := M.literal (some .fp32) .real (101.955872807799)
  let CONST036 := M.literal (some .fp32) .real (-8.67152307844475)
  let CONST037 := M.literal (some .fp32) .real (3.46860923137790)
  let CONST038 := M.literal (some .fp32) .real (-88.2963759165686)
  let CONST039 := M.literal (some .fp32) .real (-83.2466215530696)
  let CONST040 := M.literal (some .fp32) .real (-69.8044142586986)
  let CONST041 := M.literal (some .fp32) .real (-50.9779364038993)
  let CONST042 := M.literal (some .fp32) .real (-50.9779364038993)
  let CONST043 := M.literal (some .fp32) .real (-46.5362761724657)
  let CONST044 := M.literal (some .fp32) .real (-44.1481879582843)
  let CONST045 := M.literal (some .fp32) .real (-41.6233107765348)
  let CONST046 := M.literal (some .fp32) .real (-38.5356977359954)
  let CONST047 := M.literal (some .fp32) .real (-38.5356977359954)
  let CONST048 := M.literal (some .fp32) .real (-33.1662479035540)
  let CONST049 := M.literal (some .fp32) .real (-33.9852909359329)
  let CONST050 := M.literal (some .fp32) .real (6.42261628933257)
  let CONST051 := M.literal (some .fp32) .real (-33.9852909359329)
  let CONST052 := M.literal (some .fp32) .real (-29.4321253055229)
  let CONST053 := M.literal (some .fp32) .real (-27.7488738510232)
  let CONST054 := M.literal (some .fp32) .real (-20.8116553882674)
  let CONST055 := M.literal (some .fp32) .real (-19.2678488679977)
  let CONST056 := M.literal (some .fp32) .real (-19.2678488679977)
  let CONST057 := M.literal (some .fp32) .real (-16.9926454679664)
  let CONST058 := M.literal (some .fp32) .real (-16.9926454679664)
  let CONST059 := M.literal (some .fp32) .real (-13.8744369255116)
  let CONST060 := M.literal (some .fp32) .real (-16.5831239517770)
  let CONST061 := M.literal (some .fp32) .real (-8.49632273398321)
  let CONST062 := M.literal (some .fp32) .real (-6.93721846275580)
  let CONST063 := M.literal (some .fp32) .real (-5.20291384706685)
  let CONST064 := M.literal (some .fp32) .real (-3.46860923137790)
  let EXACT000 := M.literal (some .fp32) .real (11.63406904311645)
  let EXACT001 := M.literal (some .fp32) .real (-69.8044142586987)
  let EXACT002 := M.literal (some .fp32) .real (-88.2963759165687)
  let EXACT003 := M.literal (some .fp32) .real (46.5362761724656)
  let EXACT004 := M.literal (some .fp32) .real (-46.5362761724658)
  let EXACT005 := M.literal (some .fp32) .real (8.67152307844475)
  let EXACT006 := M.literal (some .fp32) .real (-50.9779364038992)
  let EXACT007 := M.literal (some .fp32) .real (8.0282703616657)
  let EXACT008 := M.literal (some .fp32) .real (24.87468592766548)
  let EXACT009 := M.literal (some .fp32) .real (24.8746859276654)
  let EXACT010 := M.literal (some .fp32) .real (-33.9852909359328)
  let EXACT011 := M.literal (some .fp32) .real (-33.98529093593284)
  let EXACT012 := M.literal (some .fp32) .real (29.43212530552288)
  let EXACT013 := M.literal (some .fp32) .real (101.9558728077987)
  let EXACT014 := M.literal (some .fp32) .real (51.3809303146604)
  let EXACT015 := M.literal (some .fp32) .real (51.3809303146604)
  let EXACT016 := M.literal (some .fp32) .real (46.5362761724656)
  let EXACT017 := M.literal (some .fp32) .real (-46.5362761724658)
  let EXACT018 := M.literal (some .fp32) .real (-88.2963759165687)
  let EXACT019 := M.literal (some .fp32) .real (11.63406904311645)
  let EXACT020 := M.literal (some .fp32) .real (-69.8044142586987)
  let EXACT021 := M.literal (some .fp32) .real (-50.9779364038992)
  let EXACT022 := M.literal (some .fp32) .real (6.42261628933256)
  let EXACT023 := M.literal (some .fp32) .real (24.8746859276654)
  let EXACT024 := M.literal (some .fp32) .real (24.87468592766548)
  let EXACT025 := M.literal (some .fp32) .real (8.0282703616657)
  let EXACT026 := M.literal (some .fp32) .real (-33.9852909359328)
  let EXACT027 := M.literal (some .fp32) .real (-33.98529093593284)
  let EXACT028 := M.literal (some .fp32) .real (29.43212530552288)
  let VAR06 := (mul (mul (mul x x) x) x)
  let VAR07 := (mul (mul x x) x)
  let VAR08 := (mul x x)
  let VAR15 := (mul (mul (mul y y) y) y)
  let VAR16 := (mul (mul y y) y)
  let VAR17 := (mul y y)
  let VAR24 := (mul (mul (mul z z) z) z)
  let VAR25 := (mul (mul z z) z)
  let VAR26 := (mul z z)
  let g_x := initial 0
  let g_y := initial 1
  let g_z := initial 2
  let g_x := add g_x (add (add (add (add (add (add (add (add (add (add (mul g_0 (add (add (mul EXACT000 VAR06) (mul CONST009 VAR24)) (mul (mul EXACT001 VAR08) VAR26))) (mul (mul g_1 y) (sub (mul (mul EXACT002 VAR08) z) (mul CONST052 VAR25)))) (mul g_10 (add (mul (mul EXACT003 VAR07) z) (mul (mul EXACT004 VAR25) x)))) (mul g_2 (add (sub (add (mul (mul CONST001 VAR08) (add (mul CONST059 VAR17) (mul CONST064 VAR26))) (mul EXACT005 VAR06)) (mul (mul CONST045 VAR17) VAR26)) (mul CONST063 VAR24)))) (mul g_3 (add (sub (mul (mul (mul EXACT006 VAR08) y) z) (mul (mul CONST049 VAR16) z)) (mul (mul CONST057 VAR25) y)))) (mul g_4 (add (add (add (add (mul CONST000 VAR24) (mul (mul CONST001 VAR08) (add (mul CONST002 VAR26) (mul CONST055 VAR17)))) (mul EXACT007 VAR06)) (mul CONST010 VAR15)) (mul (mul CONST056 VAR17) VAR26)))) (mul g_5 (add (mul (mul CONST048 VAR16) x) (mul y (add (mul EXACT008 VAR07) (mul (mul EXACT009 VAR26) x)))))) (mul g_6 (add (mul (mul CONST005 VAR25) x) (mul z (add (mul CONST004 VAR07) (mul (mul CONST046 VAR17) x)))))) (mul g_7 (sub (mul (mul EXACT010 VAR16) x) (mul (mul EXACT011 VAR07) y)))) (mul g_8 (add (mul (mul CONST008 VAR25) x) (mul z (sub (mul (mul CONST039 VAR17) x) (mul CONST054 VAR07)))))) (mul (mul g_9 y) (add (mul EXACT012 VAR07) (mul (mul CONST038 VAR26) x))))
  let g_y := add g_y (add (add (add (add (add (add (add (add (mul g_1 (sub (mul (mul CONST052 VAR07) z) (mul (mul CONST052 VAR25) x))) (mul g_2 (add (mul (mul (mul (neg CONST039) VAR26) x) y) (mul (mul CONST053 VAR07) y)))) (mul g_3 (add (mul (mul CONST058 VAR07) z) (mul x (add (mul (mul EXACT013 VAR17) z) (mul CONST057 VAR25)))))) (mul g_4 (add (mul (mul CONST047 VAR07) y) (mul x (add (mul EXACT014 VAR16) (mul (mul CONST046 VAR26) y)))))) (mul g_5 (sub (add (add (add (mul (mul CONST001 VAR17) (add (mul CONST060 VAR08) (mul CONST060 VAR26))) (mul CONST011 VAR06)) (mul CONST012 VAR24)) (mul (mul CONST014 VAR08) VAR26)) (mul CONST060 VAR15)))) (mul g_6 (add (mul (mul CONST046 VAR25) y) (mul z (add (mul EXACT015 VAR16) (mul (mul CONST046 VAR08) y)))))) (mul g_7 (add (sub (mul (mul CONST001 VAR17) (sub (mul CONST057 VAR08) (mul CONST057 VAR26))) (mul CONST061 VAR06)) (mul CONST061 VAR24)))) (mul g_8 (add (mul (mul CONST021 VAR25) y) (mul (mul (mul CONST039 VAR08) y) z)))) (mul g_9 (add (add (mul CONST027 VAR06) (mul CONST027 VAR24)) (mul (mul CONST044 VAR08) VAR26))))
  let g_z := add g_z (add (add (add (add (add (add (add (add (add (add (mul g_0 (add (mul (mul EXACT016 VAR25) x) (mul (mul EXACT017 VAR07) z))) (mul (mul g_1 y) (add (mul (mul (neg EXACT018) VAR26) x) (mul CONST052 VAR07)))) (mul g_10 (add (add (mul CONST009 VAR06) (mul EXACT019 VAR24)) (mul (mul EXACT020 VAR08) VAR26)))) (mul g_2 (add (mul (mul CONST062 VAR07) z) (mul x (add (mul (mul (neg CONST039) VAR17) z) (mul CONST054 VAR25)))))) (mul g_3 (add (mul (mul CONST058 VAR07) y) (mul x (sub (mul (mul EXACT021 VAR26) y) (mul CONST049 VAR16)))))) (mul g_4 (add (mul (mul CONST005 VAR07) z) (mul x (add (mul (mul CONST046 VAR17) z) (mul EXACT022 VAR25)))))) (mul g_5 (add (mul (mul CONST048 VAR16) z) (mul y (add (mul (mul EXACT023 VAR08) z) (mul EXACT024 VAR25)))))) (mul g_6 (add (add (add (add (mul (mul CONST001 VAR26) (add (mul CONST002 VAR08) (mul CONST056 VAR17))) (mul CONST003 VAR06)) (mul EXACT025 VAR24)) (mul CONST017 VAR15)) (mul (mul CONST056 VAR08) VAR17)))) (mul g_7 (add (mul (mul (neg EXACT026) VAR16) z) (mul (mul EXACT027 VAR25) y)))) (mul g_8 (sub (add (add (mul (mul CONST001 VAR26) (add (mul CONST018 VAR17) (mul CONST037 VAR08))) (mul CONST036 VAR24)) (mul (mul CONST045 VAR08) VAR17)) (mul CONST063 VAR06)))) (mul (mul g_9 y) (add (mul EXACT028 VAR25) (mul (mul CONST038 VAR08) z))))
  ![g_x,g_y,g_z]

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Reference
