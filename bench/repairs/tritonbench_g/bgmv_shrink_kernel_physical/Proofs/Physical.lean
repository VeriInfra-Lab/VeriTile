import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Native
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Bounds
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Native
open Contract NativeGraph VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

theorem Shared.TensorWindow.active_byte {width : Nat} (a : Shared.TensorWindow t width) (off byte : Nat)
    (active : off<t.numel) (byteInside : byte<width) :
    a.backing.covers a.window.owner (a.window.baseByte+width*off+byte) ∧
    a.window.baseByte+width*off+byte<2^64 := by
  have count : off+1≤t.numel := by omega
  have mul:=Nat.mul_le_mul_left width count
  have hi:=a.upper
  rw [a.extent] at hi
  have low:=a.lower
  have repr:=a.backing.representable
  refine ⟨⟨a.backingOwner.symm,by omega,?_⟩,?_⟩ <;> nlinarith

theorem Shared.TensorWindow.pointer_word {width : Nat} (a : Shared.TensorWindow t width) (off : Nat)
    (active : off<t.numel) (positive : 0<width) :
    (BitVec.ofNat 64 a.window.baseByte+BitVec.ofNat 64 off*BitVec.ofNat 64 width).toNat=
      t.dataPtr.toNat+width*off := by
  have inside:=(a.active_byte off 0 active positive).2
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by simpa [Nat.mul_comm] using inside)]
  rw [a.pointer,Nat.mul_comm]

theorem output_span (ok : Admitted x w y i s indices) (a : Binding x w y i s)
    (c : Cell (call x w y i s indices ok)) (owner ptr : Nat)
    (covered : (a.output.window.key (address (call x w y i s indices ok) c)).covers owner ptr) :
    a.output.backing.covers owner ptr := by
  rcases covered with ⟨ownerEq,lo,hi⟩
  simp only [Allocation.key] at ownerEq lo hi
  have cap : address (call x w y i s indices ok) c<y.numel := Bounds.output_inside ok c.1 c.2
  have inside:=a.output.active_byte _ (ptr-(a.output.window.baseByte+4*address (call x w y i s indices ok) c))
    cap (by omega)
  convert inside.1 using 1 <;> omega

theorem metadata_span (ok : Admitted x w y i s indices) (a : Binding x w y i s)
    (b : Fin (call x w y i s indices ok).batches) (byte : Fin (Shared.indexBytes i)) :
    a.indices.backing.covers a.indices.window.owner
      (i.dataPtr.toNat+Shared.indexBytes i*b.val+byte.val) ∧
    i.dataPtr.toNat+Shared.indexBytes i*b.val+byte.val<2^64 := by
  simpa only [a.indices.pointer] using a.indices.active_byte b.val byte.val b.isLt byte.isLt

theorem active_addresses (ok : Admitted x w y i s indices) (a : Binding x w y i s)
    (p : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.Program (call x w y i s indices ok)) (n : Fin (call x w y i s indices ok).N)
    (active : indices p.batch.val≠-1) (cycle e : Nat) (hk : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank p.split.val cycle e<rank w) :
    let h:=call x w y i s indices ok
    let slot:=(indices p.batch.val).toNat
    let k:=VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank p.split.val cycle e
    (BitVec.ofNat 64 a.input.window.baseByte+Bounds.inputWord h p cycle e*BitVec.ofNat 64 2).toNat=
      x.dataPtr.toNat+2*VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.inputOffset h p.batch.val k ∧
    (BitVec.ofNat 64 a.weights.window.baseByte+Bounds.weightWord h p slot n.val cycle e*BitVec.ofNat 64 2).toNat=
      w.dataPtr.toNat+2*VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.weightOffset h slot n.val k ∧
    (BitVec.ofNat 64 a.output.window.baseByte+Bounds.outputWord h p n.val*BitVec.ofNat 64 4).toNat=
      y.dataPtr.toNat+4*VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.outputOffset h p.batch.val n.val := by
  dsimp only
  have selected:=((selected_valid x w y i s indices ok p.batch.val n.val p.batch.isLt n.isLt).resolve_left active).2
  obtain ⟨wx,ww,wy⟩:=Bounds.word_offsets ok p (indices p.batch.val).toNat n.val cycle e
  rw [wx,ww,wy]
  exact ⟨a.input.pointer_word _ (Bounds.input_inside ok p.batch _ hk) (by decide),
    a.weights.pointer_word _ (Bounds.weight_inside ok _ selected n _ hk) (by decide),
    a.output.pointer_word _ (Bounds.output_inside ok p.batch n) (by decide)⟩

theorem output_address (ok : Admitted x w y i s indices) (a : Binding x w y i s)
    (p : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.Program (call x w y i s indices ok)) (n : Fin (call x w y i s indices ok).N) :
    let h:=call x w y i s indices ok
    (BitVec.ofNat 64 a.output.window.baseByte+Bounds.outputWord h p n.val*BitVec.ofNat 64 4).toNat=
      y.dataPtr.toNat+4*VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.outputOffset h p.batch.val n.val := by
  dsimp only
  rw [(Bounds.word_offsets ok p 0 n.val 0 0).2.2]
  exact a.output.pointer_word _ (Bounds.output_inside ok p.batch n) (by decide)

/-- Physical byte separation follows from checked numeric spans, never StorageImpl identity. -/
theorem backing_separate {firstWidth secondWidth : Nat}
    (a : Shared.TensorWindow t firstWidth) (b : Shared.TensorWindow u secondWidth)
    (separate : ByteSeparate t u) (owner ptr : Nat)
    (first : a.backing.covers owner ptr) : ¬b.backing.covers owner ptr := by
  rcases first with ⟨_,alo,ahi⟩
  rintro ⟨_,blo,bhi⟩
  simp only [a.backingStart,a.backingBytes] at alo ahi
  simp only [b.backingStart,b.backingBytes] at blo bhi
  rcases separate with azero|bzero|before|after <;> omega

theorem caller_frame (binding : CompilerBinding plans S) (I : Instructions)
    (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (a : Binding x w y i s) (before : Raw.State)
    (run : Run plans (call x w y i s indices ok) orders a before) (owner ptr : Nat)
    (caller : a.input.backing.covers owner ptr ∨ a.weights.backing.covers owner ptr ∨ a.indices.backing.covers owner ptr) :
    (invoke binding I x w y i s indices ok orders a before run).bytes owner ptr=before owner ptr := by
  have checks:=(python_checks_iff x w y i s indices).mp ok
  apply source_frame
  intro c _ covered
  have output:=output_span ok a c owner ptr covered
  have hb : c.1.val < i.numel := c.1.isLt
  have hn : c.2.val < width w := c.2.isLt
  rcases checks.separate with empty|empty|⟨sx,sw,si⟩
  · omega
  · omega
  · rcases caller with input|weight|index
    · exact backing_separate a.output a.input sx owner ptr output input
    · exact backing_separate a.output a.weights sw owner ptr output weight
    · exact backing_separate a.output a.indices si owner ptr output index

/-- Any subset/order of issued atomic writes has the destination frame even
when the failed launch supplies no valid numerical result. -/
theorem failure_frame (h : Call) (a : Binding x w y i s) (before : Raw.State)
    (issued : List (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Event h)) (values : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Event h → Raw.Word)
    (active : ∀ e∈issued,(data a before).indices e.1.val≠-1) (owner ptr : Nat)
    (outside : ∀ c : Cell h,(data a before).indices c.1.val≠-1 → ¬(a.output.window.key (address h c)).covers owner ptr) :
    stores Raw.ops (fun e => a.output.window.key (address h (e.1,e.2.2))) values issued before owner ptr=before owner ptr := by
  apply bytes_frame Raw.ops
  intro e member
  exact outside (e.1,e.2.2) (active e member)

theorem empty (binding : CompilerBinding plans S) (I : Instructions)
    (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h) (a : Binding x w y i s) (before : Raw.State)
    (run : Run plans h orders a before) (empty : h.batches=0 ∨ h.N=0) :
    sourceBytes binding I h orders a before run=before := by
  have noCells : cells h=[] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro c _
    have hb:=c.1.isLt
    have hn:=c.2.isLt
    rcases empty with hb'|hn' <;> omega
  simp [sourceBytes,jobs,noCells,stores]

/-- Every prefix or interleaving of actual destinations preserves the complete
read backing. Consequently later splits reload the same X/W and metadata bytes. -/
theorem issued_caller_frame (ok : Admitted x w y i s indices) (a : Binding x w y i s)
    (before : Raw.State)
    (issued : List (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Event (call x w y i s indices ok)))
    (values : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Event (call x w y i s indices ok) → Raw.Word)
    (owner ptr : Nat)
    (caller : a.input.backing.covers owner ptr ∨ a.weights.backing.covers owner ptr ∨ a.indices.backing.covers owner ptr) :
    stores Raw.ops (fun e => a.output.window.key (address (call x w y i s indices ok) (e.1,e.2.2)))
      values issued before owner ptr=before owner ptr := by
  apply bytes_frame Raw.ops
  intro e _ covered
  have output := output_span ok a (e.1,e.2.2) owner ptr covered
  have hb : e.1.val < i.numel := e.1.isLt
  have hn : e.2.2.val < width w := e.2.2.isLt
  obtain ⟨sx,sw,si⟩ := launch_separate ok (by omega) (by omega)
  rcases caller with input|weight|index
  · exact backing_separate a.output a.input sx owner ptr output input
  · exact backing_separate a.output a.weights sw owner ptr output weight
  · exact backing_separate a.output a.indices si owner ptr output index

#axiomsClean issued_caller_frame
#axiomsClean backing_separate
#axiomsClean output_span
#axiomsClean metadata_span
#axiomsClean active_addresses
#axiomsClean output_address
#axiomsClean caller_frame
#axiomsClean failure_frame
#axiomsClean empty
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Native
