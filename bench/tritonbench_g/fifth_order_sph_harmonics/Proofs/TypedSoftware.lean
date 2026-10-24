import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedReadback
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
/-- All source constexpr coefficients, encoded once in their declared fp32
profile. Each decimal-to-word result is checked below by the bit rounder. -/
def coefficient (x : ℝ) : Option Word :=
  if x = (1.73430461568895 : ℝ) then some (FP.Value.ofBits .fp32 0x3fddfdb2) else
  if x = (2.32681380862329 : ℝ) then some (FP.Value.ofBits .fp32 0x4014ea84) else
  if x = (1.60565407233314 : ℝ) then some (FP.Value.ofBits .fp32 0x3fcd8613) else
  if x = (3.21130814466628 : ℝ) then some (FP.Value.ofBits .fp32 0x404d8613) else
  if x = (3.31662479035540 : ℝ) then some (FP.Value.ofBits .fp32 0x40544395) else
  if x = (6.21867148191637 : ℝ) then some (FP.Value.ofBits .fp32 0x40c6ff5b) else
  if x = (11.6340690431164 : ℝ) then some (FP.Value.ofBits .fp32 0x413a2526) else
  if x = (12.8452325786651 : ℝ) then some (FP.Value.ofBits .fp32 0x414d8613) else
  if x = (12.4373429638327 : ℝ) then some (FP.Value.ofBits .fp32 0x4146ff5b) else
  if x = (13.8744369255116 : ℝ) then some (FP.Value.ofBits .fp32 0x415dfdb2) else
  if x = (33.9852909359329 : ℝ) then some (FP.Value.ofBits .fp32 0x4207f0f0) else
  if x = (7.35803132638072 : ℝ) then some (FP.Value.ofBits .fp32 0x40eb74fe) else
  if x = (-44.1481879582843 : ℝ) then some (FP.Value.ofBits .fp32 0xc23097bf) else
  if x = (-41.6233107765348 : ℝ) then some (FP.Value.ofBits .fp32 0xc2267e45) else
  if x = (-29.4321253055229 : ℝ) then some (FP.Value.ofBits .fp32 0xc1eb74fe) else
  if x = (-23.2681380862329 : ℝ) then some (FP.Value.ofBits .fp32 0xc1ba2526) else
  if x = (-19.2678488679977 : ℝ) then some (FP.Value.ofBits .fp32 0xc19a248e) else
  if x = (-16.9926454679664 : ℝ) then some (FP.Value.ofBits .fp32 0xc187f0f0) else
  if x = (-13.8744369255116 : ℝ) then some (FP.Value.ofBits .fp32 0xc15dfdb2) else
  if x = (-16.5831239517770 : ℝ) then some (FP.Value.ofBits .fp32 0xc184aa3d) else
  if x = (3.46860923137790 : ℝ) then some (FP.Value.ofBits .fp32 0x405dfdb2) else
  if x = (-8.49632273398321 : ℝ) then some (FP.Value.ofBits .fp32 0xc107f0f0) else
  if x = (-5.20291384706685 : ℝ) then some (FP.Value.ofBits .fp32 0xc0a67e45) else
  if x = (-3.46860923137790 : ℝ) then some (FP.Value.ofBits .fp32 0xc05dfdb2) else
  if x = (-1.73430461568895 : ℝ) then some (FP.Value.ofBits .fp32 0xbfddfdb2) else
  none

def software (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  { FP.Software.scalarFP32 rest cfg with
    literal := fun p d x => if FP.Software.fp32Context p d then
      (coefficient x).getD ((FP.Software.scalarFP32 rest cfg).literal p d x)
      else (FP.Software.scalarFP32 rest cfg).literal p d x }

def bitChannels (cfg : FP.Config) (x y z : Word) : Fin 11 → Word :=
  let add := FP.add cfg
  let sub := FP.sub cfg
  let mul := FP.mul cfg
  let neg := FP.Value.neg
  let CONST000 := FP.Value.ofBits .fp32 0x3fddfdb2
  let CONST001 := FP.Value.ofBits .fp32 0x4014ea84
  let CONST002 := FP.Value.ofBits .fp32 0x3fcd8613
  let CONST003 := FP.Value.ofBits .fp32 0x404d8613
  let CONST004 := FP.Value.ofBits .fp32 0x40544395
  let CONST005 := FP.Value.ofBits .fp32 0x40c6ff5b
  let CONST006 := FP.Value.ofBits .fp32 0x40c6ff5b
  let CONST007 := FP.Value.ofBits .fp32 0x3fcd8613
  let CONST009 := FP.Value.ofBits .fp32 0x413a2526
  let CONST010 := FP.Value.ofBits .fp32 0x414d8613
  let CONST011 := FP.Value.ofBits .fp32 0x4146ff5b
  let CONST012 := FP.Value.ofBits .fp32 0x414d8613
  let CONST013 := FP.Value.ofBits .fp32 0x415dfdb2
  let CONST017 := FP.Value.ofBits .fp32 0x4207f0f0
  let CONST018 := FP.Value.ofBits .fp32 0x40eb74fe
  let CONST020 := FP.Value.ofBits .fp32 0xc23097bf
  let CONST021 := FP.Value.ofBits .fp32 0xc2267e45
  let CONST022 := FP.Value.ofBits .fp32 0xc1eb74fe
  let CONST023 := FP.Value.ofBits .fp32 0xc1ba2526
  let CONST024 := FP.Value.ofBits .fp32 0xc19a248e
  let CONST025 := FP.Value.ofBits .fp32 0xc19a248e
  let CONST026 := FP.Value.ofBits .fp32 0xc187f0f0
  let CONST027 := FP.Value.ofBits .fp32 0xc187f0f0
  let CONST028 := FP.Value.ofBits .fp32 0xc15dfdb2
  let CONST029 := FP.Value.ofBits .fp32 0xc184aa3d
  let CONST030 := FP.Value.ofBits .fp32 0x405dfdb2
  let CONST031 := FP.Value.ofBits .fp32 0xc107f0f0
  let CONST032 := FP.Value.ofBits .fp32 0xc0a67e45
  let CONST033 := FP.Value.ofBits .fp32 0xc05dfdb2
  let CONST034 := FP.Value.ofBits .fp32 0xbfddfdb2
  let VAR05 := (mul (mul (mul (mul x x) x) x) x)
  let VAR06 := (mul (mul (mul x x) x) x)
  let VAR07 := (mul (mul x x) x)
  let VAR08 := (mul x x)
  let VAR14 := (mul (mul (mul (mul y y) y) y) y)
  let VAR15 := (mul (mul (mul y y) y) y)
  let VAR16 := (mul (mul y y) y)
  let VAR17 := (mul y y)
  let VAR23 := (mul (mul (mul (mul z z) z) z) z)
  let VAR24 := (mul (mul (mul z z) z) z)
  let VAR25 := (mul (mul z z) z)
  let VAR26 := (mul z z)
  let Y00 := (add (add (mul CONST001 VAR05) (mul (mul CONST009 VAR24) x)) (mul (mul CONST023 VAR07) VAR26))
  let Y01 := (mul y (sub (mul (mul CONST022 VAR07) z) (mul (mul CONST022 VAR25) x)))
  let Y02 := (add (add (mul CONST000 VAR05) (mul VAR07 (add (mul CONST028 VAR17) (mul CONST033 VAR26)))) (mul x (add (mul (mul (neg CONST021) VAR17) VAR26) (mul CONST032 VAR24))))
  let Y03 := (add (mul (mul (mul CONST027 VAR07) y) z) (mul x (add (mul (mul CONST017 VAR16) z) (mul (mul CONST026 VAR25) y))))
  let Y04 := (add (add (mul CONST002 VAR05) (mul VAR07 (add (mul CONST003 VAR26) (mul CONST025 VAR17)))) (mul x (add (add (mul CONST002 VAR24) (mul CONST010 VAR15)) (mul (mul CONST024 VAR17) VAR26))))
  let Y05 := (add (add (mul CONST004 VAR14) (mul VAR16 (add (mul CONST029 VAR08) (mul CONST029 VAR26)))) (mul y (add (add (mul CONST005 VAR06) (mul CONST006 VAR24)) (mul (mul CONST011 VAR08) VAR26))))
  let Y06 := (add (add (mul CONST002 VAR23) (mul VAR25 (add (mul CONST003 VAR08) (mul CONST024 VAR17)))) (mul z (add (add (mul CONST007 VAR06) (mul CONST012 VAR15)) (mul (mul CONST024 VAR08) VAR17))))
  let Y07 := (add (mul VAR16 (sub (mul CONST026 VAR08) (mul CONST026 VAR26))) (mul y (add (mul (neg CONST031) VAR06) (mul CONST031 VAR24))))
  let Y08 := (add (add (mul CONST034 VAR23) (mul VAR25 (add (mul CONST013 VAR17) (mul CONST030 VAR08)))) (mul z (sub (mul (mul CONST021 VAR08) VAR17) (mul CONST032 VAR06))))
  let Y09 := (mul y (add (add (mul CONST018 VAR06) (mul CONST018 VAR24)) (mul (mul CONST020 VAR08) VAR26)))
  let Y10 := (add (add (mul CONST001 VAR23) (mul (mul CONST009 VAR06) z)) (mul (mul CONST023 VAR08) VAR25))
  fun c => match c.val with
    | 0 => Y00
    | 1 => Y01
    | 2 => Y02
    | 3 => Y03
    | 4 => Y04
    | 5 => Y05
    | 6 => Y06
    | 7 => Y07
    | 8 => Y08
    | 9 => Y09
    | _ => Y10

theorem channels_software (rest : Algebra Word) (cfg : FP.Config) (x y z : Word) :
    channels (software rest cfg) x y z = bitChannels cfg x y z := by
  norm_num [channels, bitChannels, software, coefficient,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl

/-- These rational constants round to the listed binary32 encodings, with no
empirical or real-to-machine equality assumption. -/
theorem coefficient_words :
    (FP.round .fp32 .ieeeValue (1.73430461568895 : Rat)).bits = BitVec.ofNat 32 0x3fddfdb2 ∧
    (FP.round .fp32 .ieeeValue (2.32681380862329 : Rat)).bits = BitVec.ofNat 32 0x4014ea84 ∧
    (FP.round .fp32 .ieeeValue (1.60565407233314 : Rat)).bits = BitVec.ofNat 32 0x3fcd8613 ∧
    (FP.round .fp32 .ieeeValue (3.21130814466628 : Rat)).bits = BitVec.ofNat 32 0x404d8613 ∧
    (FP.round .fp32 .ieeeValue (3.31662479035540 : Rat)).bits = BitVec.ofNat 32 0x40544395 ∧
    (FP.round .fp32 .ieeeValue (6.21867148191637 : Rat)).bits = BitVec.ofNat 32 0x40c6ff5b ∧
    (FP.round .fp32 .ieeeValue (11.6340690431164 : Rat)).bits = BitVec.ofNat 32 0x413a2526 ∧
    (FP.round .fp32 .ieeeValue (12.8452325786651 : Rat)).bits = BitVec.ofNat 32 0x414d8613 ∧
    (FP.round .fp32 .ieeeValue (12.4373429638327 : Rat)).bits = BitVec.ofNat 32 0x4146ff5b ∧
    (FP.round .fp32 .ieeeValue (13.8744369255116 : Rat)).bits = BitVec.ofNat 32 0x415dfdb2 ∧
    (FP.round .fp32 .ieeeValue (33.9852909359329 : Rat)).bits = BitVec.ofNat 32 0x4207f0f0 ∧
    (FP.round .fp32 .ieeeValue (7.35803132638072 : Rat)).bits = BitVec.ofNat 32 0x40eb74fe ∧
    (FP.round .fp32 .ieeeValue (-44.1481879582843 : Rat)).bits = BitVec.ofNat 32 0xc23097bf ∧
    (FP.round .fp32 .ieeeValue (-41.6233107765348 : Rat)).bits = BitVec.ofNat 32 0xc2267e45 ∧
    (FP.round .fp32 .ieeeValue (-29.4321253055229 : Rat)).bits = BitVec.ofNat 32 0xc1eb74fe ∧
    (FP.round .fp32 .ieeeValue (-23.2681380862329 : Rat)).bits = BitVec.ofNat 32 0xc1ba2526 ∧
    (FP.round .fp32 .ieeeValue (-19.2678488679977 : Rat)).bits = BitVec.ofNat 32 0xc19a248e ∧
    (FP.round .fp32 .ieeeValue (-16.9926454679664 : Rat)).bits = BitVec.ofNat 32 0xc187f0f0 ∧
    (FP.round .fp32 .ieeeValue (-13.8744369255116 : Rat)).bits = BitVec.ofNat 32 0xc15dfdb2 ∧
    (FP.round .fp32 .ieeeValue (-16.5831239517770 : Rat)).bits = BitVec.ofNat 32 0xc184aa3d ∧
    (FP.round .fp32 .ieeeValue (3.46860923137790 : Rat)).bits = BitVec.ofNat 32 0x405dfdb2 ∧
    (FP.round .fp32 .ieeeValue (-8.49632273398321 : Rat)).bits = BitVec.ofNat 32 0xc107f0f0 ∧
    (FP.round .fp32 .ieeeValue (-5.20291384706685 : Rat)).bits = BitVec.ofNat 32 0xc0a67e45 ∧
    (FP.round .fp32 .ieeeValue (-3.46860923137790 : Rat)).bits = BitVec.ofNat 32 0xc05dfdb2 ∧
    (FP.round .fp32 .ieeeValue (-1.73430461568895 : Rat)).bits = BitVec.ofNat 32 0xbfddfdb2 := by
  decide +kernel

#axiomsClean channels_software
#axiomsClean coefficient_words
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
