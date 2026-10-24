import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Native

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

theorem output_inside (t : Inputs) (s : Scalars) (spans : Backings) (before : Raw.State)
    (ok : Admitted t s spans (indices t before)) (q : Nat) (hq : q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count) :
    offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q<t.y.numel := by
  have v:=valid t s spans (indices t before) ok
  have h:=coordinates (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q hq
  exact v.y _ _ h.2 h.1

theorem output_span (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (q : Nat) (hq : q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count)
    (owner address : Nat) (covered : (key t s q).covers owner address) : spans.y.covers address := by
  have c:=(python_checks_iff t s spans (indices t before)).mp ok
  have hc : t.y.isContiguous=Bool.true := (c.dense t.y (by
    simp [Contract.tensors,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.tensors])).1
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,tensorKey] at lo hi
  have inside:=(a.output.window hc).active_byte (offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q)
    (address-(t.y.dataPtr.toNat+4*offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q))
    (output_inside t s spans before ok q hq) (by omega)
  have equal : t.y.dataPtr.toNat+4*offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q+
      (address-(t.y.dataPtr.toNat+4*offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q))=address := by omega
  rw [equal] at inside
  exact inside.1

/-- Physical read-backing frames are derived from the executable byte-range
check. There is no identity-to-separation assumption on this Tensor binding. -/
theorem caller_frame (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (t : Inputs) (s : Scalars) (spans : Backings)
    (a : Binding t spans) (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (owner address : Nat)
    (read : spans.x.covers address ∨ spans.w.covers address ∨ spans.indices.covers address) :
    sourceStores rest ops schedule plans t s before owner address=before owner address := by
  have c:=(python_checks_iff t s spans (indices t before)).mp ok
  apply source_frame
  intro q hq _ covered
  have out:=output_span t s spans a before ok q hq owner address covered
  rcases read with x|w|i
  · exact disjoint_byte _ _ c.separate.1 address out x
  · exact disjoint_byte _ _ c.separate.2.1 address out w
  · exact disjoint_byte _ _ c.separate.2.2 address out i

theorem failure_frame (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (issued : List Nat) (payload : Nat→Raw.Word)
    (actual : ∀ q∈issued,q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count)
    (owner address : Nat)
    (read : spans.x.covers address ∨ spans.w.covers address ∨ spans.indices.covers address) :
    stores Raw.ops (key t s) payload issued before owner address=before owner address := by
  have c:=(python_checks_iff t s spans (indices t before)).mp ok
  apply bytes_frame Raw.ops
  intro q member covered
  have out:=output_span t s spans a before ok q (actual q member) owner address covered
  rcases read with x|w|i
  · exact disjoint_byte _ _ c.separate.1 address out x
  · exact disjoint_byte _ _ c.separate.2.1 address out w
  · exact disjoint_byte _ _ c.separate.2.2 address out i

theorem empty (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (t : Inputs) (s : Scalars) (before : Raw.State)
    (zero : t.indices.numel=0 ∨ VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w=0) :
    sourceStores rest ops schedule plans t s before=before := by
  have count : (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count=0 := by
    rcases zero with zero|zero <;>
      simp [Contract.geometry,Contract.width,
        VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry,Geometry.count,zero]
  simp [sourceStores,work,count,stores]

#axiomsClean output_inside
#axiomsClean caller_frame
#axiomsClean failure_frame
#axiomsClean empty
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
