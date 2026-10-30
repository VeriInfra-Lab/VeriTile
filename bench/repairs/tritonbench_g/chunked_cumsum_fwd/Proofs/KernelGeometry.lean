import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelMath
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelSource
import Mathlib.Data.Nat.Log

/-! Reused mathematical geometry from the historical WholeHost module,
factored here without its obsolete split-clamp execution/host imports.
These definitions describe dimensions, launch coverage and packed output
addresses only; all execution is the single Kernel.source quotation. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton
open Math
noncomputable section
set_option maxHeartbeats 1500000

def ceilDiv (n d : Nat) := (n + d - 1) / d

structure Geometry where
  batches : Nat
  sequence : Nat
  heads : Nat
  chunk : Nat
  BH : Nat
  dtBatch : Nat
  dtTime : Nat
  dtHead : Nat
  rateHead : Nat
  biasHead : Nat
  chunkPositive : 0 < chunk
  legalHeadBlock : BH ∈ [1,2,4,8,16,32,64]

def Geometry.chunks (c : Geometry) := ceilDiv c.sequence c.chunk
def Geometry.headBlocks (c : Geometry) := ceilDiv c.heads c.BH
def Geometry.BC (c : Geometry) := 2 ^ Nat.clog 2 c.chunk
def Geometry.outputSize (c : Geometry) := c.batches*c.heads*c.chunks*c.chunk

def Geometry.layout (c : Geometry) : Layout where
  batches := c.batches
  sequence := c.sequence
  heads := c.heads
  chunk := c.chunk
  BH := c.BH
  BC := c.BC
  dtBatch := c.dtBatch
  dtTime := c.dtTime
  dtHead := c.dtHead
  rateHead := c.rateHead
  biasHead := c.biasHead
  outBatch := c.heads*c.chunks*c.chunk
  outChunk := c.chunk
  outHead := c.chunks*c.chunk
  outColumn := 1
  sumBatch := c.heads*c.chunks*c.chunk
  sumChunk := c.chunk
  sumHead := c.chunks*c.chunk
  sumColumn := 1

theorem Geometry.headPositive (c : Geometry) : 0 < c.BH := by
  have h := c.legalHeadBlock
  simp only [List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with h | h | h | h | h | h | h <;> omega

theorem Geometry.chunk_le_BC (c : Geometry) : c.chunk ≤ c.BC := Nat.le_pow_clog (by omega) _

def Geometry.valid (c : Geometry) (l : Launch) : Prop :=
  l.batch < c.batches ∧ l.chunk < c.chunks ∧ l.head < c.headBlocks

def Geometry.grid (c : Geometry) : List Launch :=
  (List.range c.batches).flatMap fun b =>
    (List.range c.chunks).flatMap fun k =>
      (List.range c.headBlocks).map fun h => ⟨b,k,h⟩

@[simp] theorem Geometry.mem_grid (c : Geometry) (l : Launch) : l ∈ c.grid ↔ c.valid l := by
  rcases l with ⟨b,k,h⟩
  simp [Geometry.grid,Geometry.valid]

def packed (c : Geometry) (b h k j : Nat) := ((b*c.heads+h)*c.chunks+k)*c.chunk+j

theorem out_packed (c : Geometry) (l : Launch) (i j : Nat) :
    outAddress c.layout l i j = packed c l.batch (head c.layout l i) l.chunk j := by
  simp [outAddress,Geometry.layout,packed]
  ring

theorem sum_eq_out (c : Geometry) (l : Launch) (i j : Nat) :
    sumAddress c.layout l i j = outAddress c.layout l i j := rfl

theorem pair_unique {d a b i j : Nat} (hi : i < d) (hj : j < d)
    (he : a*d+i = b*d+j) : a = b ∧ i = j := by
  have hd : 0 < d := by omega
  have hh := congrArg (fun n => n / d) he
  simp only [Nat.mul_comm _ d, Nat.mul_add_div hd, Nat.div_eq_of_lt hi,
    Nat.div_eq_of_lt hj, Nat.add_zero] at hh
  exact ⟨hh,by rw [hh] at he; omega⟩

theorem packed_unique (c : Geometry) {b h k j b' h' k' j' : Nat}
    (hh : h < c.heads) (hk : k < c.chunks) (hj : j < c.chunk)
    (hh' : h' < c.heads) (hk' : k' < c.chunks) (hj' : j' < c.chunk)
    (he : packed c b h k j = packed c b' h' k' j') :
    b = b' ∧ h = h' ∧ k = k' ∧ j = j' := by
  obtain ⟨h1,hj⟩ := pair_unique hj hj' he
  obtain ⟨h2,hk⟩ := pair_unique hk hk' h1
  obtain ⟨hb,hh⟩ := pair_unique hh hh' h2
  exact ⟨hb,hh,hk,hj⟩

theorem output_unique (c : Geometry) (l q : Launch)
    (hl : c.valid l) (hq : c.valid q)
    (i i' : Fin c.layout.BH) (j j' : Fin c.layout.BC)
    (ha : activeOutput c.layout l i.val j.val)
    (hb : activeOutput c.layout q i'.val j'.val)
    (he : outAddress c.layout l i.val j.val = outAddress c.layout q i'.val j'.val) :
    l = q ∧ i = i' ∧ j = j' := by
  rw [out_packed,out_packed] at he
  obtain ⟨hbatch,hhead,hchunk,hj⟩ := packed_unique c ha.1 hl.2.1 ha.2 hb.1 hq.2.1 hb.2 he
  obtain ⟨hh,hi⟩ := pair_unique i.isLt i'.isLt hhead
  exact ⟨by cases l; cases q; simp_all,Fin.ext hi,Fin.ext hj⟩

theorem pair_bound {a b d i : Nat} (ha : a < b) (hi : i < d) : a*d+i < b*d := by
  have h := Nat.mul_le_mul_right d ha
  nlinarith

theorem output_bound (c : Geometry) (l : Launch) (hl : c.valid l)
    (i j : Nat) (ha : activeOutput c.layout l i j) :
    outAddress c.layout l i j < c.outputSize := by
  rw [out_packed]
  have h1 : l.batch*c.heads + head c.layout l i < c.batches*c.heads := by
    exact pair_bound hl.1 ha.1
  have h2 : (l.batch*c.heads + head c.layout l i)*c.chunks+l.chunk <
      (c.batches*c.heads)*c.chunks := pair_bound h1 hl.2.1
  exact pair_bound h2 ha.2

theorem quotient_lt_ceil {n d i : Nat} (hd : 0 < d) (hi : i < n) :
    i / d < ceilDiv n d := by
  unfold ceilDiv
  apply (Nat.lt_div_iff_mul_lt hd).2
  have hm := Nat.mod_lt i hd
  have he : i/d*d+i%d = i := by
    simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div i d
  omega

theorem cover (c : Geometry) (b h k j : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hk : k < c.chunks) (hj : j < c.chunk) :
    ∃ l ∈ c.grid, ∃ (i : Fin c.layout.BH) (t : Fin c.layout.BC),
      activeOutput c.layout l i.val t.val ∧
      l.batch = b ∧ head c.layout l i.val = h ∧ l.chunk = k ∧ t.val = j := by
  let l : Launch := ⟨b,k,h/c.BH⟩
  let i : Fin c.layout.BH := ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩
  let t : Fin c.layout.BC := ⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩
  have he : head c.layout l i.val = h := by
    simpa [head,Geometry.layout,l,i,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div h c.BH
  refine ⟨l,(c.mem_grid l).2 ⟨hb,hk,quotient_lt_ceil c.headPositive hh⟩,i,t,?_,rfl,he,rfl,rfl⟩
  refine ⟨?_,hj⟩
  change head c.layout l i.val < c.heads
  rw [he]
  exact hh


/-- The ordinary same-call input strides and packed FP32 output strides. -/
def kernel (c : Geometry) (hasBias useSoftplus : Bool) : ComputeKernel :=
  source "dt" "rate" "bias" "out" "sum" c.batches c.sequence c.heads c.chunk
    c.dtBatch c.dtTime c.dtHead c.rateHead c.biasHead
    c.layout.outBatch c.layout.outChunk c.layout.outHead c.layout.outColumn
    c.layout.sumBatch c.layout.sumChunk c.layout.sumHead c.layout.sumColumn
    useSoftplus hasBias c.BH c.BC

open FP.Structural

def tileState (l : Launch) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then l.batch else if ax=1 then l.chunk
    else if ax=2 then l.head else 0}

noncomputable def runTiles {α : Type} [Inhabited α] (M : Algebra α)
    (c : Geometry) (hasBias useSoftplus : Bool) (lower upper : α) :
    List Launch → State α → Option (State α)
  | [],s => some s
  | l::ls,s => do
      let t ← FP.Structural.exec (engine M) (kernel c hasBias useSoftplus)
        (tileState l (bindBounds s lower upper))
      runTiles M c hasBias useSoftplus lower upper ls t

def launch {α : Type} [Inhabited α] (M : Algebra α)
    (c : Geometry) (hasBias useSoftplus : Bool) (lower upper : α) (s : State α) :=
  runTiles M c hasBias useSoftplus lower upper c.grid s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
