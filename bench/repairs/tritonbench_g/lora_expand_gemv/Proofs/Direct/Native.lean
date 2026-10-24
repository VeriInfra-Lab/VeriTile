import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Checked
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Physical

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Direct.NativeCall
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000
namespace Raw
abbrev State := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.State
abbrev Word := BitVec 32
abbrev read := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.read
abbrev unpack := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.unpack
abbrev ops := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.ops
abbrev byte_of_read := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.byte_of_read
end Raw
abbrev Binding (t : Inputs) := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Binding t.x t.w t.y t.indices
abbrev ByteAllocation := VeriTile.Bench.Repairs.SwigluTriton.Memory.Allocation

def indices (a : Binding t) (before : Raw.State) :=
  VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.index before t.indices a.indices.window

def bits (a : Binding t) (before : Raw.State) (region : RegionName) (off : Nat) : Raw.Word :=
  if region="X" then Raw.read before (a.input.window.key off)
  else if region="W" then VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.half before
    a.weights.window.owner (a.weights.window.baseByte+2*off)
  else Raw.read before (a.output.window.key off)

def execution (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  execute rest plans (geometry t s) (List.range (geometry t s).count) (entry (geometry t s) (indices a before))
def finalMemory (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  (execution rest plans t s a before).getD (entry (geometry t s) (indices a before))

def reference (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (b n : Nat) :=
  Native.eval ops schedule (bits a before) ((resultCell plans (geometry t s) (indices a before) b n).read .real)

theorem complete (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (indices a before)) :
    execution rest plans t s a before=some (finalMemory rest plans t s a before) ∧
    (∀ b n,b<t.indices.numel → n<width t.w →
      finalMemory rest plans t s a before "Y" (yoff (geometry t s) b n)=resultCell plans (geometry t s) (indices a before) b n) ∧
    (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<t.indices.numel → n<width t.w → off≠yoff (geometry t s) b n) →
      finalMemory rest plans t s a before r off=entry (geometry t s) (indices a before) r off) := by
  obtain ⟨final,run,read,frame⟩:=whole_source rest plans (geometry t s) (indices a before)
    (valid t s (indices a before) ok) (List.range (geometry t s).count) (List.Perm.refl _)
  have success : execution rest plans t s a before=some final := run
  have same : finalMemory rest plans t s a before=final := by simp [finalMemory,success]
  rw [same]
  exact ⟨success,read,frame⟩

def work (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  (List.range (geometry t s).count).filter (fun q => decide (indices a before (q/(geometry t s).N)≠-1))
def key (t : Inputs) (s : Scalars) (a : Binding t) (q : Nat) := a.output.window.key (offset (geometry t s) q)
def word (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (q : Nat) :=
  Native.eval ops schedule (bits a before)
    ((finalMemory rest plans t s a before "Y" (offset (geometry t s) q)).read .real)

/-- Byte interpretation of scalar stores from actual final symbolic source
memory. No tensor copyback stage or independent-result implementation oracle. -/
def sourceStores (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) : Raw.State :=
  stores Raw.ops (key t s a) (word rest ops schedule plans t s a before) (work t s a before) before

theorem source_read (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (q : Nat) (hq : q<(geometry t s).count) (active : indices a before (q/(geometry t s).N)≠-1) :
    Raw.read (sourceStores rest ops schedule plans t s a before) (key t s a q)=word rest ops schedule plans t s a before q := by
  apply readback Raw.ops _ _ _ before q (List.mem_filter.mpr ⟨List.mem_range.mpr hq,by simpa using active⟩)
  · intro other member equal
    have addr:=congrArg Key.address equal
    have same : offset (geometry t s) other=offset (geometry t s) q := by simp only [key,Allocation.key] at addr;omega
    simp only [word,same]
  · intro other member different
    apply Allocation.indices_disjoint
    intro same
    exact different (congrArg a.output.window.key same.symm)

theorem source_frame (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (owner address : Nat)
    (outside : ∀ q,q<(geometry t s).count → indices a before (q/(geometry t s).N)≠-1 → ¬(key t s a q).covers owner address) :
    sourceStores rest ops schedule plans t s a before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro q member
  have h:=List.mem_filter.mp member
  exact outside q (List.mem_range.mp h.1) (by simpa using h.2)

theorem whole (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (indices a before)) :
    execution rest plans t s a before=some (finalMemory rest plans t s a before) ∧
    (∀ b n,b<t.indices.numel → n<width t.w → indices a before b≠-1 → ∀ byte : Fin 4,
      sourceStores rest ops schedule plans t s a before a.output.window.owner
        ((a.output.window.key (yoff (geometry t s) b n)).address+byte.val)=
          Raw.unpack (reference ops schedule plans t s a before b n) byte) ∧
    (∀ owner address,(∀ q,q<(geometry t s).count → indices a before (q/(geometry t s).N)≠-1 →
      ¬(key t s a q).covers owner address) → sourceStores rest ops schedule plans t s a before owner address=before owner address) := by
  have result:=complete rest plans t s a before ok
  refine ⟨result.1,?_,source_frame rest ops schedule plans t s a before⟩
  intro b n hb hn active byte
  have covered:=coverage (geometry t s) b n hb hn
  have read:=source_read rest ops schedule plans t s a before (b*(geometry t s).N+n) covered.1 (by simpa [covered.2.1] using active)
  have read' : Raw.read (sourceStores rest ops schedule plans t s a before) (a.output.window.key (yoff (geometry t s) b n))=
    Native.eval ops schedule (bits a before) ((finalMemory rest plans t s a before "Y" (yoff (geometry t s) b n)).read .real) := by
      simpa only [key,word,offset,covered.2.1,covered.2.2] using read
  calc
    _=Raw.unpack (Raw.read (sourceStores rest ops schedule plans t s a before) (a.output.window.key (yoff (geometry t s) b n))) byte :=
      Raw.byte_of_read _ (a.output.window.key (yoff (geometry t s) b n)) byte
    _=_ := by rw [read',result.2.1 b n hb hn];rfl

#axiomsClean complete
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean whole
#specNonCircular reference avoiding [execution,finalMemory,sourceStores,execute,Branches.source,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Direct.NativeCall
