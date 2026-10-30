import bench.tritonbench_g.token_softmax_bloom.IO

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open SoftmaxIO
noncomputable section
set_option maxHeartbeats 2000000
variable {α β : Type} [Inhabited β]

theorem start_read (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s) (r : Fin work.length) :
    startIndex p (Grid.state s.mem (work.get r)) = start p work vals r :=
  binding (0 : Fin 3) (r,()) trivial

theorem length_read (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s) (r : Fin work.length) :
    count p (Grid.state s.mem (work.get r)) = SoftmaxIO.length p work vals r :=
  binding (1 : Fin 3) (r,()) trivial

theorem payload_read (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s)
    (r : Fin work.length) (i : Fin p.width) (active : i.val < SoftmaxIO.length p work vals r) :
    (s.mem p.input (inputOffset p work vals r i.val)).read .fp32 =
      encode (vals (2 : Fin 3) (r,i,())) :=
  binding (2 : Fin 3) (r,i,()) active

theorem offset_read (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s)
    (r : Fin work.length) (i : TileIndex [p.width]) :
    Grid.offset p s.mem (work.get r) i = SoftmaxIO.outputOffset p work vals r i.1.val := by
  dsimp only [Grid.offset, VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.outputOffset]
  rw [start_read M encode p observe work vals s binding r]
  rfl

theorem unique (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s)
    (h : SoftmaxIO.Unique p work vals) : Grid.Unique p s.mem work := by
  rintro ⟨pid, i⟩ live ⟨other, j⟩ otherLive same
  have li := (Grid.mem_events p s.mem work (pid,i)).mp live
  have lj := (Grid.mem_events p s.mem work (other,j)).mp otherLive
  obtain ⟨r,hr⟩ := List.mem_iff_get.mp li.1
  obtain ⟨q,hq⟩ := List.mem_iff_get.mp lj.1
  change work.get r = pid at hr
  change work.get q = other at hq
  subst pid
  subst other
  have ai : i.1.val < SoftmaxIO.length p work vals r := by
    simpa only [Grid.active, length_read M encode p observe work vals s binding r] using li.2
  have aj : j.1.val < SoftmaxIO.length p work vals q := by
    simpa only [Grid.active, length_read M encode p observe work vals s binding q] using lj.2
  have offsets := same
  rw [offset_read M encode p observe work vals s binding r,
    offset_read M encode p observe work vals s binding q] at offsets
  obtain ⟨pidEq,laneEq⟩ := h r q i.1 j.1 ai aj offsets
  exact Prod.ext pidEq (Prod.ext laneEq (Subsingleton.elim _ _))

def RowBounds (p : Parameters) (base : Grid.Memory β) (pid : Grid.PID)
    (bounds : RegionName → Nat) : Prop :=
  pid.1*p.startStride < bounds (Region.cast p.start) ∧
  pid.1*p.lengthStride < bounds (Region.cast p.length) ∧
  (∀ i : Fin p.width, i.val < count p (Grid.state base pid) →
    pid.2*p.inputHeadStride+(startIndex p (Grid.state base pid)+i.val)*p.inputTokenStride < bounds p.input) ∧
  (∀ i : Fin p.width, i.val < count p (Grid.state base pid) →
    pid.2*p.outputHeadStride+(startIndex p (Grid.state base pid)+i.val)*p.outputTokenStride < bounds p.output)

theorem row_bounds (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s)
    (bounds : RegionName → Nat) (inside : (io M encode p observe work).InBounds bounds vals)
    (pid : Grid.PID) (live : pid ∈ work) : RowBounds p s.mem pid bounds := by
  obtain ⟨r,hr⟩ := List.mem_iff_get.mp live
  subst pid
  refine ⟨inside.2.1 (0 : Fin 3) (r,()) trivial,
    inside.2.1 (1 : Fin 3) (r,()) trivial, ?_, ?_⟩
  · intro i active
    rw [length_read M encode p observe work vals s binding r] at active
    rw [start_read M encode p observe work vals s binding r]
    exact inside.2.1 (2 : Fin 3) (r,i,()) active
  · intro i active
    rw [length_read M encode p observe work vals s binding r] at active
    rw [start_read M encode p observe work vals s binding r]
    exact inside.2.2.1 (0 : Fin 1) (r,i,()) active

#axiomsClean unique
#axiomsClean row_bounds
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOBinding
