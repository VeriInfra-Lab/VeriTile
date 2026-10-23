import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Physical
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeVocabulary

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot
open VeriTile.Triton FP.Structural NativeGraph Physical
noncomputable section

/-- Little-endian four-byte raw load. The same physical byte function supplies
all read-only tensor roles, including overlapping views and storage offsets. -/
def load32 (mem : Memory) (a : Nat) : Word :=
  BitVec.ofNat 32 ((mem a).toNat+256*(mem (a+1)).toNat+
    65536*(mem (a+2)).toNat+16777216*(mem (a+3)).toNat)
def metadataRegions (c : Host.Call) := [c.regions.start,c.regions.sequence,c.regions.prompt]

def initial (c : Host.Call) (B : Native.Bytes extent) (mem : Memory) : State Expr where
  mem := fun rg off => if rg∈metadataRegions c then
    .mk .int (load32 mem (B.base rg+4*off)).toInt
    else .mk .real (.leaf (load32 mem (B.base rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := c.grid.toNumPids
  undef := fun d _ _ => defaultValue d

/-- These are the host .tolist observations, checked nonnegative before launch. -/
structure Metadata (c : Host.Call) (B : Native.Bytes extent) (mem : Memory) : Prop where
  start : ∀ b,b<c.batches → (load32 mem (B.base c.regions.start+4*b)).toInt=(c.start b : Int)
  sequence : ∀ b,b<c.batches → (load32 mem (B.base c.regions.sequence+4*b)).toInt=((c.length b+c.prompt b : Nat) : Int)
  prompt : ∀ b,b<c.batches → (load32 mem (B.base c.regions.prompt+4*b)).toInt=(c.prompt b : Int)

theorem metadata (c : Host.Call) (B : Native.Bytes extent) (mem : Memory) (hm : Metadata c B mem) :
    FPSuccess.Metadata c (initial c B mem) := by
  refine ⟨?_,?_,?_⟩ <;> intro b hb
  · simpa [initial,metadataRegions] using hm.start b hb
  · simpa [initial,metadataRegions] using hm.sequence b hb
  · simpa [initial,metadataRegions] using hm.prompt b hb

theorem floating (c : Host.Call) (B : Native.Bytes extent) (mem : Memory)
    (rg : RegionName) (off : Nat) (hr : rg∉metadataRegions c) :
    ((initial c B mem).mem rg off).read .real=.leaf (load32 mem (B.base rg+4*off)) := by
  simp [initial,hr]

theorem good (c : Host.Call) (B : Native.Bytes extent) (mem : Memory) :
    GoodMemory (initial c B mem) := by
  intro rg off
  simp only [initial]
  split <;> simp [Cell.read,defaultValue,Good]

theorem overlapping_reads (c : Host.Call) (B : Native.Bytes extent) (mem : Memory)
    (r q : RegionName) (i j : Nat) (hr : r∉metadataRegions c) (hq : q∉metadataRegions c)
    (he : B.base r+4*i=B.base q+4*j) :
    ((initial c B mem).mem r i).read .real=((initial c B mem).mem q j).read .real := by
  rw [floating c B mem r i hr,floating c B mem q j hq,he]

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot
