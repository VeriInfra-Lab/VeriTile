import bench.tritonbench_g.flash_decode2_llama.Contract
import bench.tritonbench_g.flash_decode2_phi.Proofs.Reference
import bench.tritonbench_g.flash_decode2_phi.Proofs.Native
import bench.tritonbench_g.flash_decode2_phi.Proofs.Coordinates

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.KernelModel
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

abbrev rows (c : Call) := Checked.batches c.val * Checked.heads c.val
abbrev size (c : Call) := rows c * Checked.width c.val

def run {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (order : List Nat) (before : State α) : Option (State α) :=
  Grid.source A c.val.masked (Checked.batches c.val) (Checked.heads c.val)
    (Checked.capacity c.val) (Checked.width c.val) c.val.blockSeq (Checked.block c.val) order before

theorem source (c : Call) :
    kernel c.val.masked (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val)
      c.val.blockSeq (Checked.block c.val) =
    llama (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val)
      c.val.blockSeq (Checked.block c.val) := by
  simp only [kernel,c.property]; rfl

theorem block_eq (c : Call) : Checked.block c.val = Checked.width c.val := by
  simp [Checked.block,c.property]

theorem typed {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (order : List Nat) (all : order.Perm (List.range (rows c))) (before : State α) :
    ∃ after,run A c order before = some after ∧
      (∀ row (k : TileIndex [Checked.block c.val]), row < rows c →
        after.mem "O" (row*Checked.width c.val+lane k) = .mk .real
          (Grid.value A before c.val.masked (Checked.batches c.val) (Checked.heads c.val)
            (Checked.capacity c.val) (Checked.width c.val) c.val.blockSeq (Checked.block c.val) row k)) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ size c ≤ i → after.mem r i = before.mem r i) := by
  obtain ⟨after,executed,memory⟩ := Grid.source_complete A c.val.masked
    (Checked.batches c.val) (Checked.heads c.val) (Checked.capacity c.val)
    (Checked.width c.val) c.val.blockSeq (Checked.block c.val) order before
  have domain : c.val.masked = Bool.true ∨ Checked.block c.val = Checked.width c.val :=
    Or.inr (block_eq c)
  refine ⟨after,executed,?_,?_⟩
  · intro row k hr
    rw [memory]
    exact Grid.grid_readback A _ _ _ _ _ _ _ domain order before row
      (all.mem_iff.mpr (List.mem_range.mpr hr)) k (by rw [←block_eq c]; exact k.1.isLt)
  · intro r i outside
    rw [memory]
    apply Grid.grid_frame A _ _ _ _ _ _ _ domain order before r i
    rcases outside with outside | outside
    · exact Or.inl outside
    · right
      intro row member d hd same
      have hr := List.mem_range.mp (all.mem_iff.mp member)
      have bound : row*Checked.width c.val+d < size c := by unfold size; nlinarith
      omega

theorem empty {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (before : State α) (h : Checked.batches c.val = 0 ∨ Checked.heads c.val = 0) :
    run A c (List.range (rows c)) before = some before := by
  have zero : rows c = 0 := by rcases h with h|h <;> simp [rows,h]
  simp [run,zero,Grid.source]

theorem coverage (c : Call) (b h : Nat) (hb : b < Checked.batches c.val)
    (hh : h < Checked.heads c.val) :
    b*Checked.heads c.val+h ∈ List.range (rows c) ∧
    (b*Checked.heads c.val+h)/Checked.heads c.val=b ∧
    (b*Checked.heads c.val+h)%Checked.heads c.val=h :=
  Grid.grid_coverage _ _ b h hb hh

def iterations (c : Call) (lengths : Nat → Int) (row : Nat) : Nat :=
  ((lengths (row/Checked.heads c.val)).toNat+c.val.blockSeq-1)/c.val.blockSeq

theorem count_bound {α : Type} [Inhabited α] (c : Call) (lengths : Nat → Int)
    (h : InputContract c lengths) (before : State α)
    (binding : ∀ b,(before.mem "L" b).read .nat = (lengths b).toNat)
    (row : Nat) (hr : row < rows c) :
    count (Grid.setup (Checked.batches c.val) (Checked.heads c.val) row before) c.val.blockSeq =
      iterations c lengths row ∧
    0 < iterations c lengths row ∧ iterations c lengths row ≤ Checked.capacity c.val := by
  have hh : 0 < Checked.heads c.val := by unfold rows at hr; nlinarith
  have hb : row/Checked.heads c.val < Checked.batches c.val :=
    (Nat.div_lt_iff_lt_mul hh).mpr hr
  have active : Checked.active c.val := ⟨Nat.lt_of_le_of_lt (Nat.zero_le _) hb,hh⟩
  obtain ⟨positive,capacity,_⟩ := h.contents active _ (List.mem_range.mpr hb)
  have positive' : 0 < (lengths (row/Checked.heads c.val)).toNat := by omega
  have capacity' : (lengths (row/Checked.heads c.val)).toNat ≤ Checked.capacity c.val*c.val.blockSeq := by omega
  have qpositive := h.metadata.2.2.2.2.2.2.2.1.2.2.2.1
  refine ⟨count_eq _ _ _ (binding _) positive',?_,?_⟩
  · exact ceil_positive _ _ positive' qpositive
  · exact ceil_capacity _ _ _ qpositive capacity'

/-- Independent stage-2 summary formula; stage-1 raw-token provenance is outside this API. -/
def realTarget (c : Call) (lengths : Nat → Int) (log mid : Nat → ℝ) (row d : Nat) : ℝ :=
  Reference.target (fun j => log (row*Checked.capacity c.val+j))
    (fun j d => mid (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+d))
    (iterations c lengths row) d

#axiomsClean typed
#axiomsClean count_bound
#axiomsClean source
#specNonCircular realTarget avoiding [run,Grid.source,values,Reference.recurrence]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.KernelModel
