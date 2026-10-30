import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Arithmetic
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedSoftware

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Software
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
abbrev Word := FP.Value .fp32
abbrev forward := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.bitChannels
abbrev forwardBackend := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.software

/-- Only literals actually present in corrected backward. Each exact rational
rounding is kernel-checked below. Unused literals delegate to the base algebra. -/
def coefficient (x : ℝ) : Option Word :=
  if x=(1.60565407233314 : ℝ) then some (FP.Value.ofBits .fp32 0x3fcd8613) else
  if x=(3.00000000000000 : ℝ) then some (FP.Value.ofBits .fp32 0x40400000) else
  if x=(3.21130814466628 : ℝ) then some (FP.Value.ofBits .fp32 0x404d8613) else
  if x=(6.42261628933256 : ℝ) then some (FP.Value.ofBits .fp32 0x40cd8613) else
  if x=(8.67152307844476 : ℝ) then some (FP.Value.ofBits .fp32 0x410abe8f) else
  if x=(8.02827036166571 : ℝ) then some (FP.Value.ofBits .fp32 0x410073cc) else
  if x=(6.93721846275580 : ℝ) then some (FP.Value.ofBits .fp32 0x40ddfdb2) else
  if x=(11.6340690431164 : ℝ) then some (FP.Value.ofBits .fp32 0x413a2526) else
  if x=(12.8452325786651 : ℝ) then some (FP.Value.ofBits .fp32 0x414d8613) else
  if x=(6.21867148191637 : ℝ) then some (FP.Value.ofBits .fp32 0x40c6ff5b) else
  if x=(12.4373429638327 : ℝ) then some (FP.Value.ofBits .fp32 0x4146ff5b) else
  if x=(13.8744369255116 : ℝ) then some (FP.Value.ofBits .fp32 0x415dfdb2) else
  if x=(24.8746859276655 : ℝ) then some (FP.Value.ofBits .fp32 0x41c6ff5b) else
  if x=(27.7488738510232 : ℝ) then some (FP.Value.ofBits .fp32 0x41ddfdb2) else
  if x=(29.4321253055229 : ℝ) then some (FP.Value.ofBits .fp32 0x41eb74fe) else
  if x=(7.35803132638072 : ℝ) then some (FP.Value.ofBits .fp32 0x40eb74fe) else
  if x=(46.5362761724657 : ℝ) then some (FP.Value.ofBits .fp32 0x423a2526) else
  if x=(51.3809303146605 : ℝ) then some (FP.Value.ofBits .fp32 0x424d8613) else
  if x=(101.955872807799 : ℝ) then some (FP.Value.ofBits .fp32 0x42cbe968) else
  if x=(-8.67152307844475 : ℝ) then some (FP.Value.ofBits .fp32 0xc10abe8f) else
  if x=(3.46860923137790 : ℝ) then some (FP.Value.ofBits .fp32 0x405dfdb2) else
  if x=(-88.2963759165686 : ℝ) then some (FP.Value.ofBits .fp32 0xc2b097bf) else
  if x=(-83.2466215530696 : ℝ) then some (FP.Value.ofBits .fp32 0xc2a67e45) else
  if x=(-69.8044142586986 : ℝ) then some (FP.Value.ofBits .fp32 0xc28b9bdc) else
  if x=(-50.9779364038993 : ℝ) then some (FP.Value.ofBits .fp32 0xc24be968) else
  if x=(-46.5362761724657 : ℝ) then some (FP.Value.ofBits .fp32 0xc23a2526) else
  if x=(-44.1481879582843 : ℝ) then some (FP.Value.ofBits .fp32 0xc23097bf) else
  if x=(-41.6233107765348 : ℝ) then some (FP.Value.ofBits .fp32 0xc2267e45) else
  if x=(-38.5356977359954 : ℝ) then some (FP.Value.ofBits .fp32 0xc21a248e) else
  if x=(-33.1662479035540 : ℝ) then some (FP.Value.ofBits .fp32 0xc204aa3d) else
  if x=(-33.9852909359329 : ℝ) then some (FP.Value.ofBits .fp32 0xc207f0f0) else
  if x=(6.42261628933257 : ℝ) then some (FP.Value.ofBits .fp32 0x40cd8613) else
  if x=(-29.4321253055229 : ℝ) then some (FP.Value.ofBits .fp32 0xc1eb74fe) else
  if x=(-27.7488738510232 : ℝ) then some (FP.Value.ofBits .fp32 0xc1ddfdb2) else
  if x=(-20.8116553882674 : ℝ) then some (FP.Value.ofBits .fp32 0xc1a67e45) else
  if x=(-19.2678488679977 : ℝ) then some (FP.Value.ofBits .fp32 0xc19a248e) else
  if x=(-16.9926454679664 : ℝ) then some (FP.Value.ofBits .fp32 0xc187f0f0) else
  if x=(-13.8744369255116 : ℝ) then some (FP.Value.ofBits .fp32 0xc15dfdb2) else
  if x=(-16.5831239517770 : ℝ) then some (FP.Value.ofBits .fp32 0xc184aa3d) else
  if x=(-8.49632273398321 : ℝ) then some (FP.Value.ofBits .fp32 0xc107f0f0) else
  if x=(-6.93721846275580 : ℝ) then some (FP.Value.ofBits .fp32 0xc0ddfdb2) else
  if x=(-5.20291384706685 : ℝ) then some (FP.Value.ofBits .fp32 0xc0a67e45) else
  if x=(-3.46860923137790 : ℝ) then some (FP.Value.ofBits .fp32 0xc05dfdb2) else
  if x=(11.63406904311645 : ℝ) then some (FP.Value.ofBits .fp32 0x413a2526) else
  if x=(-69.8044142586987 : ℝ) then some (FP.Value.ofBits .fp32 0xc28b9bdc) else
  if x=(-88.2963759165687 : ℝ) then some (FP.Value.ofBits .fp32 0xc2b097bf) else
  if x=(46.5362761724656 : ℝ) then some (FP.Value.ofBits .fp32 0x423a2526) else
  if x=(-46.5362761724658 : ℝ) then some (FP.Value.ofBits .fp32 0xc23a2526) else
  if x=(8.67152307844475 : ℝ) then some (FP.Value.ofBits .fp32 0x410abe8f) else
  if x=(-50.9779364038992 : ℝ) then some (FP.Value.ofBits .fp32 0xc24be968) else
  if x=(8.0282703616657 : ℝ) then some (FP.Value.ofBits .fp32 0x410073cc) else
  if x=(24.87468592766548 : ℝ) then some (FP.Value.ofBits .fp32 0x41c6ff5b) else
  if x=(24.8746859276654 : ℝ) then some (FP.Value.ofBits .fp32 0x41c6ff5b) else
  if x=(-33.9852909359328 : ℝ) then some (FP.Value.ofBits .fp32 0xc207f0f0) else
  if x=(-33.98529093593284 : ℝ) then some (FP.Value.ofBits .fp32 0xc207f0f0) else
  if x=(29.43212530552288 : ℝ) then some (FP.Value.ofBits .fp32 0x41eb74fe) else
  if x=(101.9558728077987 : ℝ) then some (FP.Value.ofBits .fp32 0x42cbe968) else
  if x=(51.3809303146604 : ℝ) then some (FP.Value.ofBits .fp32 0x424d8613) else
  none

def backend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  { FP.Software.scalarFP32 rest cfg with
    literal:=fun p d x=>if FP.Software.fp32Context p d then
      (coefficient x).getD ((FP.Software.scalarFP32 rest cfg).literal p d x)
      else (FP.Software.scalarFP32 rest cfg).literal p d x }

def backward (cfg : FP.Config) (x y z : Word)
    (g : Fin 11 → Word) (initial : Fin 3 → Word) : Fin 3 → Word :=
  let add := FP.add cfg
  let sub := FP.sub cfg
  let mul := FP.mul cfg
  let neg := FP.Value.neg
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
  let CONST000 := FP.Value.ofBits .fp32 0x3fcd8613
  let CONST001 := FP.Value.ofBits .fp32 0x40400000
  let CONST002 := FP.Value.ofBits .fp32 0x404d8613
  let CONST003 := FP.Value.ofBits .fp32 0x3fcd8613
  let CONST004 := FP.Value.ofBits .fp32 0x40cd8613
  let CONST005 := FP.Value.ofBits .fp32 0x40cd8613
  let CONST006 := FP.Value.ofBits .fp32 0x410abe8f
  let CONST007 := FP.Value.ofBits .fp32 0x410073cc
  let CONST008 := FP.Value.ofBits .fp32 0x40ddfdb2
  let CONST009 := FP.Value.ofBits .fp32 0x413a2526
  let CONST010 := FP.Value.ofBits .fp32 0x414d8613
  let CONST011 := FP.Value.ofBits .fp32 0x40c6ff5b
  let CONST012 := FP.Value.ofBits .fp32 0x40c6ff5b
  let CONST014 := FP.Value.ofBits .fp32 0x4146ff5b
  let CONST017 := FP.Value.ofBits .fp32 0x414d8613
  let CONST018 := FP.Value.ofBits .fp32 0x415dfdb2
  let CONST019 := FP.Value.ofBits .fp32 0x41c6ff5b
  let CONST020 := FP.Value.ofBits .fp32 0x41c6ff5b
  let CONST021 := FP.Value.ofBits .fp32 0x41ddfdb2
  let CONST024 := FP.Value.ofBits .fp32 0x41eb74fe
  let CONST027 := FP.Value.ofBits .fp32 0x40eb74fe
  let CONST029 := FP.Value.ofBits .fp32 0x423a2526
  let CONST030 := FP.Value.ofBits .fp32 0x424d8613
  let CONST031 := FP.Value.ofBits .fp32 0x424d8613
  let CONST034 := FP.Value.ofBits .fp32 0x42cbe968
  let CONST036 := FP.Value.ofBits .fp32 0xc10abe8f
  let CONST037 := FP.Value.ofBits .fp32 0x405dfdb2
  let CONST038 := FP.Value.ofBits .fp32 0xc2b097bf
  let CONST039 := FP.Value.ofBits .fp32 0xc2a67e45
  let CONST040 := FP.Value.ofBits .fp32 0xc28b9bdc
  let CONST041 := FP.Value.ofBits .fp32 0xc24be968
  let CONST042 := FP.Value.ofBits .fp32 0xc24be968
  let CONST043 := FP.Value.ofBits .fp32 0xc23a2526
  let CONST044 := FP.Value.ofBits .fp32 0xc23097bf
  let CONST045 := FP.Value.ofBits .fp32 0xc2267e45
  let CONST046 := FP.Value.ofBits .fp32 0xc21a248e
  let CONST047 := FP.Value.ofBits .fp32 0xc21a248e
  let CONST048 := FP.Value.ofBits .fp32 0xc204aa3d
  let CONST049 := FP.Value.ofBits .fp32 0xc207f0f0
  let CONST050 := FP.Value.ofBits .fp32 0x40cd8613
  let CONST051 := FP.Value.ofBits .fp32 0xc207f0f0
  let CONST052 := FP.Value.ofBits .fp32 0xc1eb74fe
  let CONST053 := FP.Value.ofBits .fp32 0xc1ddfdb2
  let CONST054 := FP.Value.ofBits .fp32 0xc1a67e45
  let CONST055 := FP.Value.ofBits .fp32 0xc19a248e
  let CONST056 := FP.Value.ofBits .fp32 0xc19a248e
  let CONST057 := FP.Value.ofBits .fp32 0xc187f0f0
  let CONST058 := FP.Value.ofBits .fp32 0xc187f0f0
  let CONST059 := FP.Value.ofBits .fp32 0xc15dfdb2
  let CONST060 := FP.Value.ofBits .fp32 0xc184aa3d
  let CONST061 := FP.Value.ofBits .fp32 0xc107f0f0
  let CONST062 := FP.Value.ofBits .fp32 0xc0ddfdb2
  let CONST063 := FP.Value.ofBits .fp32 0xc0a67e45
  let CONST064 := FP.Value.ofBits .fp32 0xc05dfdb2
  let EXACT000 := FP.Value.ofBits .fp32 0x413a2526
  let EXACT001 := FP.Value.ofBits .fp32 0xc28b9bdc
  let EXACT002 := FP.Value.ofBits .fp32 0xc2b097bf
  let EXACT003 := FP.Value.ofBits .fp32 0x423a2526
  let EXACT004 := FP.Value.ofBits .fp32 0xc23a2526
  let EXACT005 := FP.Value.ofBits .fp32 0x410abe8f
  let EXACT006 := FP.Value.ofBits .fp32 0xc24be968
  let EXACT007 := FP.Value.ofBits .fp32 0x410073cc
  let EXACT008 := FP.Value.ofBits .fp32 0x41c6ff5b
  let EXACT009 := FP.Value.ofBits .fp32 0x41c6ff5b
  let EXACT010 := FP.Value.ofBits .fp32 0xc207f0f0
  let EXACT011 := FP.Value.ofBits .fp32 0xc207f0f0
  let EXACT012 := FP.Value.ofBits .fp32 0x41eb74fe
  let EXACT013 := FP.Value.ofBits .fp32 0x42cbe968
  let EXACT014 := FP.Value.ofBits .fp32 0x424d8613
  let EXACT015 := FP.Value.ofBits .fp32 0x424d8613
  let EXACT016 := FP.Value.ofBits .fp32 0x423a2526
  let EXACT017 := FP.Value.ofBits .fp32 0xc23a2526
  let EXACT018 := FP.Value.ofBits .fp32 0xc2b097bf
  let EXACT019 := FP.Value.ofBits .fp32 0x413a2526
  let EXACT020 := FP.Value.ofBits .fp32 0xc28b9bdc
  let EXACT021 := FP.Value.ofBits .fp32 0xc24be968
  let EXACT022 := FP.Value.ofBits .fp32 0x40cd8613
  let EXACT023 := FP.Value.ofBits .fp32 0x41c6ff5b
  let EXACT024 := FP.Value.ofBits .fp32 0x41c6ff5b
  let EXACT025 := FP.Value.ofBits .fp32 0x410073cc
  let EXACT026 := FP.Value.ofBits .fp32 0xc207f0f0
  let EXACT027 := FP.Value.ofBits .fp32 0xc207f0f0
  let EXACT028 := FP.Value.ofBits .fp32 0x41eb74fe
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

theorem backward_software (rest : Algebra Word) (cfg : FP.Config) (x y z : Word)
    (g : Fin 11 → Word) (initial : Fin 3 → Word) :
    Reference.backwardValues (backend rest cfg) x y z g initial=backward cfg x y z g initial := by
  norm_num [Reference.backwardValues,backward,backend,coefficient,
    FP.Software.scalarFP32,FP.Software.fp32Context]

theorem coefficient_words :
    (FP.round .fp32 .ieeeValue (1.60565407233314 : Rat)).bits=BitVec.ofNat 32 0x3fcd8613 ∧
    (FP.round .fp32 .ieeeValue (3.00000000000000 : Rat)).bits=BitVec.ofNat 32 0x40400000 ∧
    (FP.round .fp32 .ieeeValue (3.21130814466628 : Rat)).bits=BitVec.ofNat 32 0x404d8613 ∧
    (FP.round .fp32 .ieeeValue (6.42261628933256 : Rat)).bits=BitVec.ofNat 32 0x40cd8613 ∧
    (FP.round .fp32 .ieeeValue (8.67152307844476 : Rat)).bits=BitVec.ofNat 32 0x410abe8f ∧
    (FP.round .fp32 .ieeeValue (8.02827036166571 : Rat)).bits=BitVec.ofNat 32 0x410073cc ∧
    (FP.round .fp32 .ieeeValue (6.93721846275580 : Rat)).bits=BitVec.ofNat 32 0x40ddfdb2 ∧
    (FP.round .fp32 .ieeeValue (11.6340690431164 : Rat)).bits=BitVec.ofNat 32 0x413a2526 ∧
    (FP.round .fp32 .ieeeValue (12.8452325786651 : Rat)).bits=BitVec.ofNat 32 0x414d8613 ∧
    (FP.round .fp32 .ieeeValue (6.21867148191637 : Rat)).bits=BitVec.ofNat 32 0x40c6ff5b ∧
    (FP.round .fp32 .ieeeValue (12.4373429638327 : Rat)).bits=BitVec.ofNat 32 0x4146ff5b ∧
    (FP.round .fp32 .ieeeValue (13.8744369255116 : Rat)).bits=BitVec.ofNat 32 0x415dfdb2 ∧
    (FP.round .fp32 .ieeeValue (24.8746859276655 : Rat)).bits=BitVec.ofNat 32 0x41c6ff5b ∧
    (FP.round .fp32 .ieeeValue (27.7488738510232 : Rat)).bits=BitVec.ofNat 32 0x41ddfdb2 ∧
    (FP.round .fp32 .ieeeValue (29.4321253055229 : Rat)).bits=BitVec.ofNat 32 0x41eb74fe ∧
    (FP.round .fp32 .ieeeValue (7.35803132638072 : Rat)).bits=BitVec.ofNat 32 0x40eb74fe ∧
    (FP.round .fp32 .ieeeValue (46.5362761724657 : Rat)).bits=BitVec.ofNat 32 0x423a2526 ∧
    (FP.round .fp32 .ieeeValue (51.3809303146605 : Rat)).bits=BitVec.ofNat 32 0x424d8613 ∧
    (FP.round .fp32 .ieeeValue (101.955872807799 : Rat)).bits=BitVec.ofNat 32 0x42cbe968 ∧
    (FP.round .fp32 .ieeeValue (-8.67152307844475 : Rat)).bits=BitVec.ofNat 32 0xc10abe8f ∧
    (FP.round .fp32 .ieeeValue (3.46860923137790 : Rat)).bits=BitVec.ofNat 32 0x405dfdb2 ∧
    (FP.round .fp32 .ieeeValue (-88.2963759165686 : Rat)).bits=BitVec.ofNat 32 0xc2b097bf ∧
    (FP.round .fp32 .ieeeValue (-83.2466215530696 : Rat)).bits=BitVec.ofNat 32 0xc2a67e45 ∧
    (FP.round .fp32 .ieeeValue (-69.8044142586986 : Rat)).bits=BitVec.ofNat 32 0xc28b9bdc ∧
    (FP.round .fp32 .ieeeValue (-50.9779364038993 : Rat)).bits=BitVec.ofNat 32 0xc24be968 ∧
    (FP.round .fp32 .ieeeValue (-46.5362761724657 : Rat)).bits=BitVec.ofNat 32 0xc23a2526 ∧
    (FP.round .fp32 .ieeeValue (-44.1481879582843 : Rat)).bits=BitVec.ofNat 32 0xc23097bf ∧
    (FP.round .fp32 .ieeeValue (-41.6233107765348 : Rat)).bits=BitVec.ofNat 32 0xc2267e45 ∧
    (FP.round .fp32 .ieeeValue (-38.5356977359954 : Rat)).bits=BitVec.ofNat 32 0xc21a248e ∧
    (FP.round .fp32 .ieeeValue (-33.1662479035540 : Rat)).bits=BitVec.ofNat 32 0xc204aa3d ∧
    (FP.round .fp32 .ieeeValue (-33.9852909359329 : Rat)).bits=BitVec.ofNat 32 0xc207f0f0 ∧
    (FP.round .fp32 .ieeeValue (6.42261628933257 : Rat)).bits=BitVec.ofNat 32 0x40cd8613 ∧
    (FP.round .fp32 .ieeeValue (-29.4321253055229 : Rat)).bits=BitVec.ofNat 32 0xc1eb74fe ∧
    (FP.round .fp32 .ieeeValue (-27.7488738510232 : Rat)).bits=BitVec.ofNat 32 0xc1ddfdb2 ∧
    (FP.round .fp32 .ieeeValue (-20.8116553882674 : Rat)).bits=BitVec.ofNat 32 0xc1a67e45 ∧
    (FP.round .fp32 .ieeeValue (-19.2678488679977 : Rat)).bits=BitVec.ofNat 32 0xc19a248e ∧
    (FP.round .fp32 .ieeeValue (-16.9926454679664 : Rat)).bits=BitVec.ofNat 32 0xc187f0f0 ∧
    (FP.round .fp32 .ieeeValue (-13.8744369255116 : Rat)).bits=BitVec.ofNat 32 0xc15dfdb2 ∧
    (FP.round .fp32 .ieeeValue (-16.5831239517770 : Rat)).bits=BitVec.ofNat 32 0xc184aa3d ∧
    (FP.round .fp32 .ieeeValue (-8.49632273398321 : Rat)).bits=BitVec.ofNat 32 0xc107f0f0 ∧
    (FP.round .fp32 .ieeeValue (-6.93721846275580 : Rat)).bits=BitVec.ofNat 32 0xc0ddfdb2 ∧
    (FP.round .fp32 .ieeeValue (-5.20291384706685 : Rat)).bits=BitVec.ofNat 32 0xc0a67e45 ∧
    (FP.round .fp32 .ieeeValue (-3.46860923137790 : Rat)).bits=BitVec.ofNat 32 0xc05dfdb2 ∧
    (FP.round .fp32 .ieeeValue (11.63406904311645 : Rat)).bits=BitVec.ofNat 32 0x413a2526 ∧
    (FP.round .fp32 .ieeeValue (-69.8044142586987 : Rat)).bits=BitVec.ofNat 32 0xc28b9bdc ∧
    (FP.round .fp32 .ieeeValue (-88.2963759165687 : Rat)).bits=BitVec.ofNat 32 0xc2b097bf ∧
    (FP.round .fp32 .ieeeValue (46.5362761724656 : Rat)).bits=BitVec.ofNat 32 0x423a2526 ∧
    (FP.round .fp32 .ieeeValue (-46.5362761724658 : Rat)).bits=BitVec.ofNat 32 0xc23a2526 ∧
    (FP.round .fp32 .ieeeValue (8.67152307844475 : Rat)).bits=BitVec.ofNat 32 0x410abe8f ∧
    (FP.round .fp32 .ieeeValue (-50.9779364038992 : Rat)).bits=BitVec.ofNat 32 0xc24be968 ∧
    (FP.round .fp32 .ieeeValue (8.0282703616657 : Rat)).bits=BitVec.ofNat 32 0x410073cc ∧
    (FP.round .fp32 .ieeeValue (24.87468592766548 : Rat)).bits=BitVec.ofNat 32 0x41c6ff5b ∧
    (FP.round .fp32 .ieeeValue (24.8746859276654 : Rat)).bits=BitVec.ofNat 32 0x41c6ff5b ∧
    (FP.round .fp32 .ieeeValue (-33.9852909359328 : Rat)).bits=BitVec.ofNat 32 0xc207f0f0 ∧
    (FP.round .fp32 .ieeeValue (-33.98529093593284 : Rat)).bits=BitVec.ofNat 32 0xc207f0f0 ∧
    (FP.round .fp32 .ieeeValue (29.43212530552288 : Rat)).bits=BitVec.ofNat 32 0x41eb74fe ∧
    (FP.round .fp32 .ieeeValue (101.9558728077987 : Rat)).bits=BitVec.ofNat 32 0x42cbe968 ∧
    (FP.round .fp32 .ieeeValue (51.3809303146604 : Rat)).bits=BitVec.ofNat 32 0x424d8613 := by
  decide +kernel

#axiomsClean backward_software
#axiomsClean coefficient_words
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Software
