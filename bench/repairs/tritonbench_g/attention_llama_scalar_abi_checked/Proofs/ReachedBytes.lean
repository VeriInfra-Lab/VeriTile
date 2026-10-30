import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Values

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural
noncomputable section

@[simp] theorem pointers_region (c : Config) (s : State (Term α))
    (R : RegionName) (sz sh sr sd n : Nat) (i : TileIndex [64,c.D]) :
    (pointers c s R sz sh sr sd n i).1 = R := rfl

/-- Representation of exactly the masked K/V byte reads. It places no
condition on FP16 execution, output padding or exterior caller storage. -/
structure ReachedBytes (c : Config) (s : State (Term α)) : Prop where
  key : c.fp8=Bool.true → ∀ n (i : TileIndex [64,c.D]), n*64+i.1.val<c.N →
    Float8E5Bits.isInt8 ((s.mem c.K (pointers c s c.K c.skz c.skh c.skn c.skk n i).2).read .int)=Bool.true
  value : c.fp8=Bool.true → ∀ n (i : TileIndex [64,c.D]), n*64+i.1.val<c.N →
    Float8E5Bits.isInt8 ((s.mem c.V (pointers c s c.V c.svz c.svh c.svk c.svn n i).2).read .int)=Bool.true

theorem masked_key_cell (c : Config) (s : State (Term α)) (h : ReachedBytes c s)
    (n : Nat) (i : TileIndex [64,c.D]) (hf : c.fp8=Bool.true) :
    Float8E5Bits.isInt8 (if n*64+i.1.val<c.N then
      (s.mem c.K (pointers c s c.K c.skz c.skh c.skn c.skk n i).2).read .int else 0)=Bool.true := by
  split
  · exact h.key hf n i ‹_›
  · decide

theorem masked_value_cell (c : Config) (s : State (Term α)) (h : ReachedBytes c s)
    (n : Nat) (i : TileIndex [64,c.D]) (hf : c.fp8=Bool.true) :
    Float8E5Bits.isInt8 (if n*64+i.1.val<c.N then
      (s.mem c.V (pointers c s c.V c.svz c.svh c.svk c.svn n i).2).read .int else 0)=Bool.true := by
  split
  · exact h.value hf n i ‹_›
  · decide

theorem masked_key_bytes (c : Config) (s : State (Term α)) (h : ReachedBytes c s)
    (n : Nat) (hf : c.fp8=Bool.true) :
    (TileShape.allIndices [64,c.D]).all (fun i=>Float8E5Bits.isInt8
      (if n*64+i.1.val<c.N then
        (s.mem c.K (pointers c s c.K c.skz c.skh c.skn c.skk n i).2).read .int else 0))=Bool.true := by
  simp only [List.all_eq_true]
  exact fun i _=>masked_key_cell c s h n i hf

theorem masked_value_bytes (c : Config) (s : State (Term α)) (h : ReachedBytes c s)
    (n : Nat) (hf : c.fp8=Bool.true) :
    (TileShape.allIndices [64,c.D]).all (fun i=>Float8E5Bits.isInt8
      (if n*64+i.1.val<c.N then
        (s.mem c.V (pointers c s c.V c.svz c.svh c.svk c.svn n i).2).read .int else 0))=Bool.true := by
  simp only [List.all_eq_true]
  exact fun i _=>masked_value_cell c s h n i hf

#axiomsClean masked_key_bytes
#axiomsClean masked_value_bytes
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
