import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Checked
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Binding
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def indices (t : Inputs) (before : Raw.State) (b : Nat) : Int :=
  let address:=t.indices.dataPtr.toNat+indexBytes t.indices*b
  if t.indices.dtype=.other "int64" then
    (Raw.read before ⟨t.indices.device.index,address+4⟩ ++
      Raw.read before ⟨t.indices.device.index,address⟩).toInt
  else (Raw.read before ⟨t.indices.device.index,address⟩).toInt

def bits (t : Inputs) (before : Raw.State) (region : RegionName) (off : Nat) : Raw.Word :=
  if region="X" then Raw.read before (tensorKey t.x off)
  else if region="W" then
    (before t.w.device.index (t.w.dataPtr.toNat+2*off+1) ++
      before t.w.device.index (t.w.dataPtr.toNat+2*off)).zeroExtend 32
  else Raw.read before (tensorKey t.y off)

def execution (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :=
  execute rest plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count) (entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before))
def finalMemory (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :=
  (execution rest plans t s before).getD (entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before))

def reference (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) (b n : Nat) :=
  Native.eval ops schedule (bits t before) ((resultCell plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before) b n).read .real)

theorem complete (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) (ok : Admitted t s spans (indices t before)) :
    execution rest plans t s before=some (finalMemory rest plans t s before) ∧
    (∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w →
      finalMemory rest plans t s before "Y" (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)=resultCell plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before) b n) ∧
    (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w → off≠yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n) →
      finalMemory rest plans t s before r off=entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before) r off) := by
  obtain ⟨final,run,read,frame⟩:=whole_source rest plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before)
    (valid t s spans (indices t before) ok) (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count) (List.Perm.refl _)
  have success : execution rest plans t s before=some final := run
  have same : finalMemory rest plans t s before=final := by simp [finalMemory,success]
  rw [same]
  exact ⟨success,read,frame⟩

def work (t : Inputs) (s : Scalars) (before : Raw.State) :=
  (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count).filter (fun q => decide (indices t before (q/(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).N)≠-1))
def key (t : Inputs) (s : Scalars) (q : Nat) := tensorKey t.y (offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q)
def word (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) (q : Nat) :=
  Native.eval ops schedule (bits t before)
    ((finalMemory rest plans t s before "Y" (offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q)).read .real)

/-- Byte interpretation of scalar stores from actual final symbolic source
memory. No tensor copyback stage or independent-result implementation oracle. -/
def sourceStores (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) : Raw.State :=
  stores Raw.ops (key t s) (word rest ops schedule plans t s before) (work t s before) before

theorem source_read (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State)
    (q : Nat) (hq : q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count) (active : indices t before (q/(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).N)≠-1) :
    Raw.read (sourceStores rest ops schedule plans t s before) (key t s q)=word rest ops schedule plans t s before q := by
  apply readback Raw.ops _ _ _ before q (List.mem_filter.mpr ⟨List.mem_range.mpr hq,by simpa using active⟩)
  · intro other member equal
    have addr:=congrArg Key.address equal
    have same : offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) other=offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q := by simp only [key,tensorKey] at addr;omega
    simp only [word,same]
  · intro other member different
    have ne : offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) other≠offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q := by
      intro same
      apply different
      simp only [key,same]
    dsimp only [key,tensorKey,Key.disjoint]
    omega

theorem source_frame (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) (owner address : Nat)
    (outside : ∀ q,q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count → indices t before (q/(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).N)≠-1 → ¬(key t s q).covers owner address) :
    sourceStores rest ops schedule plans t s before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro q member
  have h:=List.mem_filter.mp member
  exact outside q (List.mem_range.mp h.1) (by simpa using h.2)

theorem whole (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) (ok : Admitted t s spans (indices t before)) :
    execution rest plans t s before=some (finalMemory rest plans t s before) ∧
    (∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w → indices t before b≠-1 → ∀ byte : Fin 4,
      sourceStores rest ops schedule plans t s before t.y.device.index
        ((tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).address+byte.val)=
          Raw.unpack (reference ops schedule plans t s before b n) byte) ∧
    (∀ owner address,(∀ q,q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count → indices t before (q/(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).N)≠-1 →
      ¬(key t s q).covers owner address) → sourceStores rest ops schedule plans t s before owner address=before owner address) := by
  have result:=complete rest plans t s before ok
  refine ⟨result.1,?_,source_frame rest ops schedule plans t s before⟩
  intro b n hb hn active byte
  have covered:=coverage (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n hb hn
  have read:=source_read rest ops schedule plans t s before (b*(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).N+n) covered.1 (by simpa [covered.2.1] using active)
  have read' : Raw.read (sourceStores rest ops schedule plans t s before) (tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n))=
    Native.eval ops schedule (bits t before) ((finalMemory rest plans t s before "Y" (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).read .real) := by
      simpa only [key,word,offset,covered.2.1,covered.2.2] using read
  calc
    _=Raw.unpack (Raw.read (sourceStores rest ops schedule plans t s before) (tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n))) byte :=
      Raw.byte_of_read _ (tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)) byte
    _=_ := by rw [read',result.2.1 b n hb hn];rfl

#axiomsClean complete
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean whole
#specNonCircular reference avoiding [execution,finalMemory,sourceStores,execute,Branches.source,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
