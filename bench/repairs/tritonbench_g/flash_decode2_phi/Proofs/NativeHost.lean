import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Native
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Host

namespace VeriTile.Bench.Repairs.FlashDecode2.Native
open VeriTile.Triton FP.Structural
noncomputable section

/-- Actual storage allocations bound to the four logical pointer arguments.
M/E may share read-only FP32 storage. The checked host excludes output aliases
and mixed INT32/FP32 aliases on a nonempty launch. -/
structure Bindings where
  lengths : RegionName
  mid : RegionName
  log : RegionName
  out : RegionName

def Bindings.at (b : Bindings) (r : RegionName) : RegionName :=
  if r="L" then b.lengths else if r="M" then b.mid else if r="E" then b.log else if r="O" then b.out else r

def bindings {α : Type} [Inhabited α] {q : Host.Request} (call : Host.Preparation α q) : Bindings :=
  ⟨call.lengths.allocation,call.mid.allocation,call.log.allocation,call.out.allocation⟩

/-- INT32 words decode in the integer channel; the signed int32 load is decoded before transport to the natural loop channel. FP32 input words
remain opaque leaves and include all bit patterns, NaNs, infinities and zeros. -/
def entry (b : Bindings) (memory : RegionName → Nat → Word) : State Term where
  mem := fun r o => if r="L" then .mk .nat (memory b.lengths o).toInt.toNat
    else .mk .real (.leaf r o (memory (b.at r) o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def value (I : Instructions) (S : Schedule) (b : Bindings)
    (memory : RegionName → Nat → Word) (q : Host.Request) (row : Nat)
    (k : TileIndex [q.g.laneBlock]) : Word :=
  eval I S (Grid.value algebra (entry b memory) q.g.masked q.g.batches q.g.heads
    q.g.capacity q.g.dimension q.g.sequenceBlock q.g.laneBlock row k)

theorem whole_native (I : Instructions) (S : Schedule) (q : Host.Request)
    (memory : RegionName → Nat → Word) (call : Host.Preparation Term q)
    (hbefore : call.before=entry (bindings call) memory) (schedule : List Nat)
    (all : schedule.Perm (List.range q.g.rows)) :
    ∃ result,Host.invoke algebra q call schedule=some result ∧ result.returned=none ∧
      result.suppliedOutput=call.out ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.g.rows=0 then none else some call.owner) ∧
      (∀ row (k : TileIndex [q.g.laneBlock]),row<q.g.rows → lane k<q.g.dimension →
        observe I S result.state "O" (row*q.g.dimension+lane k)=
          value I S (bindings call) memory q row k) ∧
      (∀ r a,r≠("O" : RegionName) ∨ q.g.size≤a → result.state.mem r a=call.before.mem r a) := by
  obtain ⟨result,he,hn,ho,hctx,hl,hv,hf⟩ := Host.whole_typed algebra q call schedule all
  refine ⟨result,he,hn,ho,hctx,hl,?_,hf⟩
  intro row k hr hk
  unfold observe
  rw [hv row k hr hk,Cell.read_same,hbefore]
  rfl

#axiomsClean whole_native
#specNonCircular entry avoiding [Host.invoke,Grid.source,FP.Structural.exec]
#specNonCircular value avoiding [Host.invoke,Grid.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FlashDecode2.Native
