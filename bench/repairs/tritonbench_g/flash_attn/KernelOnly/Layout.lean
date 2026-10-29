import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Grid
import bench.repairs.tritonbench_g.flash_attn.Proofs.Grid

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural
noncomputable section

def parameters (x : Layout) : Parameters :=
  ⟨x.q.batch,x.q.head,x.q.row,x.q.dim,x.k.batch,x.k.head,x.k.row,x.k.dim,
   x.v.batch,x.v.head,x.v.row,x.v.dim,x.o.batch,x.o.head,x.o.row,x.o.dim,
   x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim,x.shape.causal⟩

def regionName : FlashAttn.Region→RegionName
  | .q=>"Q" | .k=>"K" | .v=>"V" | .o=>"O" | .l=>"L" | .other n=>⟨"other"++toString n⟩

def sourceRegion (r : RegionName) : FlashAttn.Region :=
  if r="Q" then .q else if r="K" then .k else if r="V" then .v
  else if r="O" then .o else if r="L" then .l else .other 0

def liftMemory {α : Type} (m : Memory α) : Mem α := fun r off=>.mk .real (m (sourceRegion r) off)

def Program.Valid (x : Layout) (job : Program) : Prop :=
  job.1<(x.shape.length+127)/128 ∧ job.2<x.shape.batches*x.shape.heads

theorem program_valid (x : Layout) (job : Program) : job∈programs (parameters x) ↔ job.Valid x := by
  rcases job with ⟨row,head⟩
  simp [programs,parameters,Program.Valid]

theorem writes_footprint (x : Layout) (hs : 0<x.shape.heads) (r : RegionName) (off : Nat)
    (hw : Writes (parameters x) (programs (parameters x)) r off) :
    footprint x (sourceRegion r) off := by
  rcases hw with ⟨job,hj,h⟩
  have valid:=(program_valid x job).mp hj
  have hb : job.2/x.shape.heads<x.shape.batches := (Nat.div_lt_iff_lt_mul hs).mpr valid.2
  have hh : job.2%x.shape.heads<x.shape.heads := Nat.mod_lt _ hs
  rcases h with ⟨rfl,i,hi,rfl⟩|⟨rfl,i,hi,rfl⟩
  · refine ⟨.output (job.2/x.shape.heads) (job.2%x.shape.heads) (job.1*128+i.1.val) i.2.1.val,?_,?_⟩
    · exact ⟨hb,hh,hi,i.2.1.isLt⟩
    · simp [Work.target,sourceRegion,parameters,FlashAttn.address]
  · refine ⟨.logsum (job.2/x.shape.heads) (job.2%x.shape.heads) (job.1*128+i.1.val),⟨hb,hh,hi⟩,?_⟩
    simp [Work.target,sourceRegion,parameters,FlashAttn.lAddress]
    left
    simpa [Nat.mul_comm] using (Nat.div_add_mod job.2 x.shape.heads).symm

#axiomsClean writes_footprint
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
