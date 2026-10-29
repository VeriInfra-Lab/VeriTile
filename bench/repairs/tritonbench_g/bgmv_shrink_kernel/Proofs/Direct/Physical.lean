import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.Native
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.Bounds

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Native
open Contract NativeGraph VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

theorem output_span (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (c : Cell (call x w y i s indices ok)) (owner ptr : Nat)
    (covered : (a.output.window.key (address (call x w y i s indices ok) c)).covers owner ptr) :
    a.output.backing.covers owner ptr := by
  rcases covered with ⟨ownerEq,lo,hi⟩
  simp only [Allocation.key] at ownerEq lo hi
  have cap : address (call x w y i s indices ok) c<y.numel := Bounds.output_inside ok c.1 c.2
  have inside:=a.output.active_byte _ (ptr-(a.output.window.baseByte+4*address (call x w y i s indices ok) c))
    cap (by omega)
  convert inside.1 using 1 <;> omega

theorem metadata_span (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (b : Fin (call x w y i s indices ok).batches) (byte : Fin (Shared.indexBytes i)) :
    a.indices.backing.covers a.indices.window.owner
      (i.dataPtr.toNat+Shared.indexBytes i*b.val+byte.val) ∧
    i.dataPtr.toNat+Shared.indexBytes i*b.val+byte.val<2^64 := by
  simpa only [a.indices.pointer] using a.indices.active_byte b.val byte.val b.isLt byte.isLt

theorem active_addresses (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (p : Geometry.Program (call x w y i s indices ok)) (n : Fin (call x w y i s indices ok).N)
    (active : indices p.batch.val≠-1) (cycle e : Nat) (hk : Geometry.rank p.split.val cycle e<rank w) :
    let h:=call x w y i s indices ok
    let slot:=(indices p.batch.val).toNat
    let k:=Geometry.rank p.split.val cycle e
    (BitVec.ofNat 64 a.input.window.baseByte+Bounds.inputWord h p cycle e*BitVec.ofNat 64 2).toNat=
      x.dataPtr.toNat+2*Geometry.inputOffset h p.batch.val k ∧
    (BitVec.ofNat 64 a.weights.window.baseByte+Bounds.weightWord h p slot n.val cycle e*BitVec.ofNat 64 2).toNat=
      w.dataPtr.toNat+2*Geometry.weightOffset h slot n.val k ∧
    (BitVec.ofNat 64 a.output.window.baseByte+Bounds.outputWord h p n.val*BitVec.ofNat 64 4).toNat=
      y.dataPtr.toNat+4*Geometry.outputOffset h p.batch.val n.val := by
  dsimp only
  have selected:=((selected_valid x w y i s indices ok p.batch.val n.val p.batch.isLt n.isLt).resolve_left active).2
  obtain ⟨wx,ww,wy⟩:=Bounds.word_offsets ok p (indices p.batch.val).toNat n.val cycle e
  rw [wx,ww,wy]
  exact ⟨a.input.pointer_word _ (Bounds.input_inside ok p.batch _ hk) (by decide),
    a.weights.pointer_word _ (Bounds.weight_inside ok _ selected n _ hk) (by decide),
    a.output.pointer_word _ (Bounds.output_inside ok p.batch n) (by decide)⟩

theorem output_address (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (p : Geometry.Program (call x w y i s indices ok)) (n : Fin (call x w y i s indices ok).N) :
    let h:=call x w y i s indices ok
    (BitVec.ofNat 64 a.output.window.baseByte+Bounds.outputWord h p n.val*BitVec.ofNat 64 4).toNat=
      y.dataPtr.toNat+4*Geometry.outputOffset h p.batch.val n.val := by
  dsimp only
  rw [(Bounds.word_offsets ok p 0 n.val 0 0).2.2]
  exact a.output.pointer_word _ (Bounds.output_inside ok p.batch n) (by decide)

theorem caller_frame (binding : CompilerBinding plans S) (I : Instructions)
    (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : Workset.Orders (call x w y i s indices ok)) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans (call x w y i s indices ok) orders a before) (owner ptr : Nat)
    (caller : a.input.backing.covers owner ptr ∨ a.weights.backing.covers owner ptr ∨ a.indices.backing.covers owner ptr) :
    (invoke binding I x w y i s indices ok orders a before run).bytes owner ptr=before owner ptr := by
  have checks:=(python_checks_iff x w y i s indices).mp ok
  apply source_frame
  intro c _ covered
  have output:=output_span ok a c owner ptr covered
  rcases caller with input|weight|index
  · exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _
      (a.inputSeparate checks.separate.1.symm) owner ptr input output
  · exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _
      (a.weightSeparate checks.separate.2.1.symm) owner ptr weight output
  · exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _
      (a.indexSeparate checks.separate.2.2.symm) owner ptr index output

/-- Any subset/order of issued atomic writes has the destination frame even
when the failed launch supplies no valid numerical result. -/
theorem failure_frame (h : Call) (a : Binding x w y i) (before : Raw.State)
    (issued : List (Workset.Event h)) (values : Workset.Event h → Raw.Word)
    (active : ∀ e∈issued,(data a before).indices e.1.val≠-1) (owner ptr : Nat)
    (outside : ∀ c : Cell h,(data a before).indices c.1.val≠-1 → ¬(a.output.window.key (address h c)).covers owner ptr) :
    stores Raw.ops (fun e => a.output.window.key (address h (e.1,e.2.2))) values issued before owner ptr=before owner ptr := by
  apply bytes_frame Raw.ops
  intro e member
  exact outside (e.1,e.2.2) (active e member)

theorem empty (binding : CompilerBinding plans S) (I : Instructions)
    (h : Call) (orders : Workset.Orders h) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans h orders a before) (empty : h.batches=0 ∨ h.N=0) :
    sourceBytes binding I h orders a before run=before := by
  have noCells : cells h=[] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro c _
    have hb:=c.1.isLt
    have hn:=c.2.isLt
    rcases empty with hb'|hn' <;> omega
  simp [sourceBytes,jobs,noCells,stores]

#axiomsClean output_span
#axiomsClean metadata_span
#axiomsClean active_addresses
#axiomsClean output_address
#axiomsClean caller_frame
#axiomsClean failure_frame
#axiomsClean empty
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Native
