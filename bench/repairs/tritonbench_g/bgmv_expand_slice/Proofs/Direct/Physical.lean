import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Native
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Bounds

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native
open Contract VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000

theorem TensorWindow.active_byte {width : Nat} (a : TensorWindow t width) (off byte : Nat)
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

theorem TensorWindow.pointer_word {width : Nat} (a : TensorWindow t width) (off : Nat)
    (active : off<t.numel) (positive : 0<width) :
    (BitVec.ofNat 64 a.window.baseByte+BitVec.ofNat 64 off*BitVec.ofNat 64 width).toNat=
      t.dataPtr.toNat+width*off := by
  have inside:=(a.active_byte off 0 active positive).2
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by simpa [Nat.mul_comm] using inside)]
  rw [a.pointer,Nat.mul_comm]

theorem output_inside (ok : Admitted x w y i s indices)
    (p : Geometry.Program (call x w y i s indices ok)) :
    FPGrid.address (call x w y i s indices ok) p<y.numel := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have bound:=Geometry.active_output (call x w y i s indices ok) p
  have cap:=Nat.mul_le_mul_right (columns y) c.shapes.2.2.2.1
  rw [Bounds.matrix_numel y c.ranks.2.1]
  exact bound.trans_le cap

theorem output_span (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (p : Geometry.Program (call x w y i s indices ok)) (owner address : Nat)
    (covered : (a.output.window.key (FPGrid.address (call x w y i s indices ok) p)).covers owner address) :
    a.output.backing.covers owner address := by
  rcases covered with ⟨ownerEq,lo,hi⟩
  simp only [Allocation.key] at ownerEq lo hi
  have inside:=a.output.active_byte _ (address-(a.output.window.baseByte+4*FPGrid.address (call x w y i s indices ok) p))
    (output_inside ok p) (by omega)
  convert inside.1 using 1 <;> omega

theorem metadata_span (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (p : Geometry.Program (call x w y i s indices ok)) (byte : Fin (indexBytes i)) :
    a.indices.backing.covers a.indices.window.owner
      (i.dataPtr.toNat+indexBytes i*p.batch+byte.val) ∧
    i.dataPtr.toNat+indexBytes i*p.batch+byte.val<2^64 := by
  simpa only [a.indices.pointer] using a.indices.active_byte p.batch byte.val p.batchInside byte.isLt

/-- Direct pointers for all active source loads and output stores follow from
ordinary tensor windows and the executable shape/content checks. -/
theorem active_addresses (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i)
    (p : Geometry.Program (call x w y i s indices ok)) (active : indices p.batch≠-1)
    (k : Nat) (hk : k<rank w) :
    let h:=call x w y i s indices ok
    let slot:=(indices p.batch).toNat
    (BitVec.ofNat 64 a.input.window.baseByte+Bounds.inputWord h p k*BitVec.ofNat 64 4).toNat=
      x.dataPtr.toNat+4*Bounds.inputOffset h p k ∧
    (BitVec.ofNat 64 a.weights.window.baseByte+Bounds.weightWord h p slot 0 k*BitVec.ofNat 64 2).toNat=
      w.dataPtr.toNat+2*Bounds.weightOffset h p slot 0 k ∧
    (BitVec.ofNat 64 a.output.window.baseByte+Bounds.outputWord h p 0*BitVec.ofNat 64 4).toNat=
      y.dataPtr.toNat+4*Bounds.outputOffset h p 0 := by
  dsimp only
  have selected:=((selected_valid x w y i s indices ok p.batch p.row p.batchInside p.rowInside).resolve_left active).2
  have inside:=Bounds.active x w y i s indices ok p (indices p.batch).toNat selected k hk
  obtain ⟨wx,ww,wy⟩:=Bounds.word_offsets x w y i s indices ok p (indices p.batch).toNat 0 k
  rw [wx,ww,wy]
  exact ⟨a.input.pointer_word _ inside.1 (by decide),a.weights.pointer_word _ inside.2.1 (by decide),
    a.output.pointer_word _ inside.2.2 (by decide)⟩

theorem separate_byte (read output : Allocation) (separate : read.separate output)
    (owner address : Nat) (covered : read.covers owner address) : ¬output.covers owner address := by
  rcases separate with device|before|after <;> rcases covered with ⟨ownerEq,lo,hi⟩
  all_goals rintro ⟨otherOwner,otherLo,otherHi⟩
  · exact device (ownerEq.symm.trans otherOwner)
  · omega
  · omega

theorem caller_frame (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i) (before : Raw.State)
    (owner address : Nat) (caller : a.input.backing.covers owner address ∨
      a.weights.backing.covers owner address ∨ a.indices.backing.covers owner address) :
    (invoke plans S I binding x w y i s indices ok a before).bytes owner address=before owner address := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  apply source_frame
  intro p _ covered
  have output:=output_span ok a p owner address covered
  rcases caller with input|weight|index
  · exact separate_byte _ _ (a.inputSeparate c.separate.1.symm) owner address input output
  · exact separate_byte _ _ (a.weightSeparate c.separate.2.1.symm) owner address weight output
  · exact separate_byte _ _ (a.indexSeparate c.separate.2.2.symm) owner address index output

/-- A launch interruption can leave a prefix of actual source stores. Its
frame is exact; numerical success and exception atomicity are not assumed. -/
def interrupted (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State)
    (count : Nat) : Raw.State :=
  stores Raw.ops (fun p => a.output.window.key (FPGrid.address h p))
    (outputWord plans S I binding h a before) ((jobs h (data a before).indices).take count) before

theorem interrupted_frame (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State)
    (count owner address : Nat) (outside : ∀ p : Geometry.Program h,(data a before).indices p.batch≠-1 →
      ¬(a.output.window.key (FPGrid.address h p)).covers owner address) :
    interrupted plans S I binding h a before count owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro p member
  exact outside p (by simpa using (List.mem_filter.mp (List.mem_of_mem_take member)).2)

/-- Any issued subset/order of source destinations has the same frame, even
when a failed launch provides no correct numerical value for those stores. -/
theorem failure_frame (h : Call) (a : Binding x w y i) (before : Raw.State)
    (issued : List (Geometry.Program h)) (values : Geometry.Program h → Raw.Word)
    (active : ∀ p∈issued,(data a before).indices p.batch≠-1)
    (owner address : Nat) (outside : ∀ p : Geometry.Program h,(data a before).indices p.batch≠-1 →
      ¬(a.output.window.key (FPGrid.address h p)).covers owner address) :
    stores Raw.ops (fun p => a.output.window.key (FPGrid.address h p)) values issued before owner address=
      before owner address := by
  apply bytes_frame Raw.ops
  intro p member
  exact outside p (active p member)

theorem empty (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State)
    (empty : h.batches=0 ∨ h.N=0) : sourceStores plans S I binding h a before=before := by
  have noPrograms : Geometry.programs h=[] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro p _
    have hb:=p.batchInside
    have hr:=p.rowInside
    rcases empty with hb'|hr' <;> omega
  simp [sourceStores,jobs,noPrograms,stores]

#axiomsClean TensorWindow.active_byte
#axiomsClean TensorWindow.pointer_word
#axiomsClean output_inside
#axiomsClean metadata_span
#axiomsClean active_addresses
#axiomsClean caller_frame
#axiomsClean interrupted_frame
#axiomsClean failure_frame
#axiomsClean empty
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native
