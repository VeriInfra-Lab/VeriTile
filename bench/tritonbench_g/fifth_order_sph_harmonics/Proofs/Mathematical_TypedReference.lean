import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Mathematical_TypedSource
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
/-- Source-order operation graph on three already-loaded scalar inputs. No
real polynomial identities or reassociation are used by this specification. -/
def channels {α : Type} (M : Algebra α) (x y z : α) : Fin 11 → α :=
  let add := M.binary (some .fp32) .real .add
  let sub := M.binary (some .fp32) .real .sub
  let mul := M.binary (some .fp32) .real .mul
  let neg := M.unary (some .fp32) (.neg .real)
  let CONST000 := M.literal (some .fp32) .real (1.73430461568895)
  let CONST001 := M.literal (some .fp32) .real (2.32681380862329)
  let CONST002 := M.literal (some .fp32) .real (1.60565407233314)
  let CONST003 := M.literal (some .fp32) .real (3.21130814466628)
  let CONST004 := M.literal (some .fp32) .real (3.31662479035540)
  let CONST005 := M.literal (some .fp32) .real (6.21867148191637)
  let CONST006 := M.literal (some .fp32) .real (6.21867148191637)
  let CONST007 := M.literal (some .fp32) .real (1.60565407233314)
  let CONST009 := M.literal (some .fp32) .real (11.6340690431164)
  let CONST010 := M.literal (some .fp32) .real (12.8452325786651)
  let CONST011 := M.literal (some .fp32) .real (12.4373429638327)
  let CONST012 := M.literal (some .fp32) .real (12.8452325786651)
  let CONST013 := M.literal (some .fp32) .real (13.8744369255116)
  let CONST017 := M.literal (some .fp32) .real (33.9852909359329)
  let CONST018 := M.literal (some .fp32) .real (7.35803132638072)
  let CONST020 := M.literal (some .fp32) .real (-44.1481879582843)
  let CONST021 := M.literal (some .fp32) .real (-41.6233107765348)
  let CONST022 := M.literal (some .fp32) .real (-29.4321253055229)
  let CONST023 := M.literal (some .fp32) .real (-23.2681380862329)
  let CONST024 := M.literal (some .fp32) .real (-19.2678488679977)
  let CONST025 := M.literal (some .fp32) .real (-19.2678488679977)
  let CONST026 := M.literal (some .fp32) .real (-16.9926454679664)
  let CONST027 := M.literal (some .fp32) .real (-16.9926454679664)
  let CONST028 := M.literal (some .fp32) .real (-13.8744369255116)
  let CONST029 := M.literal (some .fp32) .real (-16.5831239517770)
  let CONST030 := M.literal (some .fp32) .real (3.46860923137790)
  let CONST031 := M.literal (some .fp32) .real (-8.49632273398321)
  let CONST032 := M.literal (some .fp32) .real (-5.20291384706685)
  let CONST033 := M.literal (some .fp32) .real (-3.46860923137790)
  let CONST034 := M.literal (some .fp32) .real (-1.73430461568895)
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

def coordOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.blockSize]) : Nat :=
  i.1.val * 3 + p.blockSize * 3 * s.pids 0

def outputOffset {α : Type} (p : Parameters) (s : State α) (c : Fin 11) (i : TileIndex [p.blockSize]) : Nat :=
  i.1.val * p.outputStride + p.blockSize * p.outputStride * s.pids 0 + p.columnOffset + c.val

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (coordinate : Fin 3) (i : TileIndex [p.blockSize]) : α :=
  M.fp32Load (if coordOffset p s i + coordinate.val < p.coordNumel then
    (s.mem p.input (coordOffset p s i + coordinate.val)).read .real
    else s.undef .real p.input (coordOffset p s i + coordinate.val))

def expected {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (c : Fin 11) (i : TileIndex [p.blockSize]) : α :=
  channels M (loaded M p s 0 i) (loaded M p s 1 i) (loaded M p s 2 i) c

end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
