import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FP4Bits
import Lean
open Lean VeriTile.Bench.TritonBenchG.Fp4ToBf16.Bits

/-- Exports all packed inputs from the existing kernel-checked table relation.
This is a CPU reference generator, not a GPU correspondence proof. -/
def main : IO Unit := do
  let rows := (List.range 256).map fun b =>
    let x : Packed := BitVec.ofNat 8 b
    Json.mkObj [("input", toJson b),
      ("bf16", toJson [(expectedBits (high x)).toNat, (expectedBits (low x)).toNat]),
      ("source_fp32", toJson [(sourceWord32 (high x)).toNat, (sourceWord32 (low x)).toNat]),
      ("software_bf16", toJson [(decodedPair x).1.bits.toNat, (decodedPair x).2.bits.toNat])]
  IO.println (Json.mkObj [("schema", toJson (1 : Nat)),
    ("theorem", toJson ("VeriTile.Bench.TritonBenchG.Fp4ToBf16.Bits.every_packed_byte" : String)),
    ("rows", toJson rows)]).compress
