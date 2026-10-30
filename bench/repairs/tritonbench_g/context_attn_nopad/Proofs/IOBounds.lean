import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.IOData
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.IOLoopSafety

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def coordinate (c : Call) (idx : GridIndex c.grid) (t : Nat) (ht : t<timeExtent c)
    (d : Fin c.q.dims) : Coordinate c :=
  (⟨(c.launch idx).batch,launch_batch_lt c idx⟩,⟨(c.launch idx).head,launch_head_lt c idx⟩,
    ⟨t,ht⟩,d,PUnit.unit)

theorem row_bound (c : Call) (idx : GridIndex c.grid) (i : Fin 128) :
    (c.launch idx).block*128+i.val<timeExtent c := by
  have := launch_block_lt c idx
  have := i.isLt
  dsimp [timeExtent]
  omega

theorem aligned_bound (c : Call) (idx : GridIndex c.grid) (n : Nat)
    (hn : n*128<((c.launch idx).block+1)*128) (j : Fin 128) :
    n*128+j.val<timeExtent c := by
  have := launch_block_lt c idx
  have := j.isLt
  dsimp [timeExtent]
  omega

theorem source_bounds (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (xs : (io c r o).Inputs) (bounds : RegionName → Nat) (hb : (io c r o).InBounds bounds xs)
    (s : State α) (metadata : FPHostSource.Metadata c r s) (idx : GridIndex c.grid) :
    IOSourceSafety.Bounds c.layout r ((FPHost.atProgram c idx s).pids 0)
      (FPReference.inputs r (FPHost.atProgram c idx s)) bounds := by
  have hs := metadata.start (c.launch idx).batch (launch_batch_lt c idx)
  have hl := metadata.length (c.launch idx).batch (launch_batch_lt c idx)
  constructor
  · exact hb.2.1 3 (⟨_,launch_batch_lt c idx⟩,PUnit.unit) trivial
  · exact hb.2.1 4 (⟨_,launch_batch_lt c idx⟩,PUnit.unit) trivial
  · intro i d on
    change (((c.launch idx).block*128+i.val : Nat) : Int)<(s.mem r.sequence (c.launch idx).batch).read .int at on
    rw [hl] at on
    have ht : (c.launch idx).block*128+i.val<c.length (c.launch idx).batch := by exact_mod_cast on
    have h := hb.2.1 0 (coordinate c idx _ (row_bound c idx i) d) ht
    change c.q.offset (c.start (c.launch idx).batch+((c.launch idx).block*128+i.val)) (c.launch idx).head d.val<bounds r.query at h
    change (((s.mem r.start (c.launch idx).batch).read .int+((c.launch idx).block*128+i.val : Nat))*c.q.rowStride+
      ((c.launch idx).head*c.q.headStride : Nat)+(d.val*c.q.dimStride : Nat)).toNat<bounds r.query
    rw [hs]
    simpa only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Tensor3.offset] using h
  · intro cur _ aligned before j d on
    obtain ⟨n,rfl⟩ := aligned
    change ((n*128 : Nat) : Int)<((FPReference.blockMask c.layout (FPReference.inputs r (FPHost.atProgram c idx s))*((c.launch idx).block+1)*128 : Nat) : Int) at before
    have hm : FPReference.blockMask c.layout (FPReference.inputs r (FPHost.atProgram c idx s)) ≤ 1 := by
      unfold FPReference.blockMask
      split <;> omega
    have hstop : n*128<((c.launch idx).block+1)*128 := by
      have hb' : n*128<FPReference.blockMask c.layout (FPReference.inputs r (FPHost.atProgram c idx s))*((c.launch idx).block+1)*128 := by exact_mod_cast before
      exact lt_of_lt_of_le hb' (Nat.mul_le_mul_right 128 (by simpa using Nat.mul_le_mul_right ((c.launch idx).block+1) hm))
    change ((n*128 : Nat) : Int)+j.val<(s.mem r.sequence (c.launch idx).batch).read .int at on
    rw [hl] at on
    have ht : n*128+j.val<c.length (c.launch idx).batch := by exact_mod_cast on
    have h := hb.2.1 1 (coordinate c idx _ (aligned_bound c idx n hstop j) d) ht
    change c.k.offset (c.start (c.launch idx).batch+(n*128+j.val)) (c.launch idx).head d.val<bounds r.key at h
    change j.val*c.k.rowStride+(c.launch idx).head*c.k.headStride+d.val*c.k.dimStride+
      (((s.mem r.start (c.launch idx).batch).read .int+(n*128 : Nat))*c.k.rowStride).toNat<bounds r.key
    rw [hs]
    rw [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    simpa [Tensor3.offset,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro cur _ aligned before j d on
    obtain ⟨n,rfl⟩ := aligned
    change ((n*128 : Nat) : Int)<((FPReference.blockMask c.layout (FPReference.inputs r (FPHost.atProgram c idx s))*((c.launch idx).block+1)*128 : Nat) : Int) at before
    have hm : FPReference.blockMask c.layout (FPReference.inputs r (FPHost.atProgram c idx s)) ≤ 1 := by
      unfold FPReference.blockMask
      split <;> omega
    have hstop : n*128<((c.launch idx).block+1)*128 := by
      have hb' : n*128<FPReference.blockMask c.layout (FPReference.inputs r (FPHost.atProgram c idx s))*((c.launch idx).block+1)*128 := by exact_mod_cast before
      exact lt_of_lt_of_le hb' (Nat.mul_le_mul_right 128 (by simpa using Nat.mul_le_mul_right ((c.launch idx).block+1) hm))
    change ((n*128 : Nat) : Int)+j.val<(s.mem r.sequence (c.launch idx).batch).read .int at on
    rw [hl] at on
    have ht : n*128+j.val<c.length (c.launch idx).batch := by exact_mod_cast on
    have h := hb.2.1 2 (coordinate c idx _ (aligned_bound c idx n hstop j) d) ht
    change c.v.offset (c.start (c.launch idx).batch+(n*128+j.val)) (c.launch idx).head d.val<bounds r.value at h
    change j.val*c.v.rowStride+(c.launch idx).head*c.v.headStride+d.val*c.v.dimStride+
      (((s.mem r.start (c.launch idx).batch).read .int+(n*128 : Nat))*c.v.rowStride).toNat<bounds r.value
    rw [hs]
    rw [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    simpa [Tensor3.offset,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro i d on
    change (((c.launch idx).block*128+i.val : Nat) : Int)<(s.mem r.sequence (c.launch idx).batch).read .int at on
    rw [hl] at on
    have ht : (c.launch idx).block*128+i.val<c.length (c.launch idx).batch := by exact_mod_cast on
    have h := hb.2.2.1 0 (coordinate c idx _ (row_bound c idx i) d) ht
    change c.o.offset (c.start (c.launch idx).batch+((c.launch idx).block*128+i.val)) (c.launch idx).head d.val<bounds r.out at h
    change (((s.mem r.start (c.launch idx).batch).read .int+((c.launch idx).block*128+i.val : Nat))*c.o.rowStride+
      ((c.launch idx).head*c.o.headStride : Nat)+(d.val*c.o.dimStride : Nat)).toNat<bounds r.out
    rw [hs]
    simpa only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Tensor3.offset] using h

#axiomsClean source_bounds
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
