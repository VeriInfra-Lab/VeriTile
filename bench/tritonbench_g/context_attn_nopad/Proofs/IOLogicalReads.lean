import bench.tritonbench_g.context_attn_nopad.Proofs.IOLogical

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open IOData
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type}

theorem query_congr (c : Call) (r : Regions) (M : Algebra α) (idx : GridIndex c.grid)
    (left right : Memory α) (same : SameReads c r left right) :
    FPReference.queryTile c.layout M (arguments c r idx left)=
      FPReference.queryTile c.layout M (arguments c r idx right) := by
  funext lane
  rcases lane with ⟨i,d,⟨⟩⟩
  simp only [FPReference.queryTile,arguments,FPReference.row,Call.layout]
  split_ifs with on
  · have ht : (c.launch idx).block*128+i.val<c.length (c.launch idx).batch := by exact_mod_cast on
    congr 1
    have h := same r.query (address c c.q (coordinate c idx _ (row_bound c idx i) d))
      ⟨coordinate c idx _ (row_bound c idx i) d,ht,Or.inl ⟨rfl,rfl⟩⟩
    simpa only [address,coordinate,Tensor3.offset,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast] using h
  · rfl

theorem key_congr (c : Call) (r : Regions) (M : Algebra α) (idx : GridIndex c.grid)
    (left right : Memory α) (same : SameReads c r left right) (n : Nat)
    (hn : n*128<((c.launch idx).block+1)*128) :
    FPReference.keyTile c.layout M (arguments c r idx left) (n*128 : Nat)=
      FPReference.keyTile c.layout M (arguments c r idx right) (n*128 : Nat) := by
  funext lane
  rcases lane with ⟨d,j,⟨⟩⟩
  simp only [FPReference.keyTile,arguments,Call.layout]
  split_ifs with on
  · have ht : n*128+j.val<c.length (c.launch idx).batch := by exact_mod_cast on
    congr 1
    rw [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    have h := same r.key (address c c.k (coordinate c idx _ (aligned_bound c idx n hn j) d))
      ⟨coordinate c idx _ (aligned_bound c idx n hn j) d,ht,Or.inr (Or.inl ⟨rfl,rfl⟩)⟩
    simpa [address,coordinate,Tensor3.offset,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · rfl

theorem value_congr (c : Call) (r : Regions) (M : Algebra α) (idx : GridIndex c.grid)
    (left right : Memory α) (same : SameReads c r left right) (n : Nat)
    (hn : n*128<((c.launch idx).block+1)*128) :
    FPReference.valueTile c.layout M (arguments c r idx left) (n*128 : Nat)=
      FPReference.valueTile c.layout M (arguments c r idx right) (n*128 : Nat) := by
  funext lane
  rcases lane with ⟨j,d,⟨⟩⟩
  simp only [FPReference.valueTile,arguments,Call.layout]
  split_ifs with on
  · have ht : n*128+j.val<c.length (c.launch idx).batch := by exact_mod_cast on
    congr 1
    rw [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    have h := same r.value (address c c.v (coordinate c idx _ (aligned_bound c idx n hn j) d))
      ⟨coordinate c idx _ (aligned_bound c idx n hn j) d,ht,Or.inr (Or.inr ⟨rfl,rfl⟩)⟩
    simpa [address,coordinate,Tensor3.offset,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · rfl

theorem advance_congr (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (left right : Memory α) (same : SameReads c r left right) (n : Nat)
    (hn : n*128<((c.launch idx).block+1)*128) (old : FPReference.Acc c.layout α) :
    FPReference.advance c.layout o c.scale (arguments c r idx left) (n*128 : Nat) old=
      FPReference.advance c.layout o c.scale (arguments c r idx right) (n*128 : Nat) old := by
  simp only [FPReference.advance,query_congr c r o.algebra idx left right same,
    key_congr c r o.algebra idx left right same n hn,value_congr c r o.algebra idx left right same n hn]
  rfl

#axiomsClean advance_congr
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
