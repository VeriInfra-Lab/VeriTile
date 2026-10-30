import bench.tritonbench_g.cross_entropy1.Proofs.Library

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor coordinateOffset)
noncomputable section

/-- Actual row-major logical enumeration through an arbitrary two-dimensional
Torch view. The result keeps both element strides and the storage offset. -/
theorem matrix_index (t : Tensor) (rows cols rs cs row col : Nat)
    (shape : t.shape=[rows,cols]) (strides : t.strides=[rs,cs])
    (hr : row<rows) (hc : col<cols) :
    t.index (row*cols+col)=t.storageOffset+row*rs+col*cs := by
  have div : (row*cols+col)/cols=row := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul,Nat.one_mul]
      omega
  have mod : (row*cols+col)%cols=col := by
    have h := Nat.div_add_mod (row*cols+col) cols
    rw [div,Nat.mul_comm cols row] at h
    omega
  simp [Tensor.index,coordinateOffset,shape,strides,div,mod,Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hc,Nat.add_assoc]

theorem vector_index (t : Tensor) (rows stride row : Nat)
    (shape : t.shape=[rows]) (strides : t.strides=[stride]) (hr : row<rows) :
    t.index row=t.storageOffset+row*stride := by
  simp [Tensor.index,coordinateOffset,shape,strides,Nat.mod_eq_of_lt hr]

/-- A unit last stride is sufficient for the kernel's ordinary row layout;
no assumption excludes zero row strides, gaps, or overlapping input rows. -/
theorem last_stride_layout (t : Tensor) (rank : t.strides.length=t.shape.length)
    (last : lastStride t=1) : RowLayout t := by
  intro rows cols shape row col hr hc
  have len : t.strides.length=2 := by simpa [shape] using rank
  rcases t with ⟨objectId,region,shape',strides,offset,neg,cuda,device,dtype⟩
  dsimp at *
  cases strides with
  | nil => simp at len
  | cons rs tail =>
    cases tail with
    | nil => simp at len
    | cons cs tail =>
      cases tail with
      | cons a rest => simp at len
      | nil =>
        have csOne : cs=1 := by simpa [lastStride] using last
        have hi := matrix_index ⟨objectId,region,shape',[rs,cs],offset,neg,cuda,device,dtype⟩
          rows cols rs cs row col shape rfl hr hc
        simpa [rowStride,csOne] using hi

theorem nextPower_positive (cols : Nat) (positive : 0<cols) : 0<nextPower cols := by
  simp only [nextPower,if_neg (Nat.ne_of_gt positive)]
  positivity

theorem nextPower_cover (cols : Nat) (positive : 0<cols) : cols≤nextPower cols := by
  simp only [nextPower,if_neg (Nat.ne_of_gt positive)]
  exact Nat.le_pow_clog (by decide) cols

theorem forward_width (cols : Nat) (positive : 0<cols) :
    0<fwdWidth cols ∧ fwdWidth cols≤65536 ∧ (cols≤65536 → cols≤fwdWidth cols) := by
  have h := nextPower_positive cols positive
  have cover := nextPower_cover cols positive
  simp only [fwdWidth]
  constructor
  · exact Nat.lt_min.mpr ⟨h,by decide⟩
  · exact ⟨Nat.min_le_right _ _,fun hc => Nat.le_min.mpr ⟨cover,hc⟩⟩

theorem backward_width (cols : Nat) (positive : 0<cols) :
    0<bwdWidth cols ∧ bwdWidth cols≤4096 := by
  have h := nextPower_positive cols positive
  exact ⟨Nat.lt_min.mpr ⟨h,by decide⟩,Nat.min_le_right _ _⟩

theorem block_count_positive (cols width : Nat) (positive : 0<cols) (widthPositive : 0<width) :
    0<Math.blocks cols width := by
  have h := Math.blocks_cover cols width widthPositive
  by_contra zero
  have count : Math.blocks cols width=0 := by omega
  rw [count,Nat.zero_mul] at h
  omega

theorem one_block (cols width : Nat) (positive : 0<cols) (covered : cols≤width) :
    Math.blocks cols width=1 := by
  unfold Math.blocks
  apply Nat.div_eq_of_lt_le
  · simp only [Nat.one_mul]
    omega
  · simp only [Nat.reduceAdd,Nat.succ_mul,Nat.one_mul]
    omega

#axiomsClean forward_width
#axiomsClean backward_width
#axiomsClean block_count_positive
#axiomsClean one_block
#axiomsClean matrix_index
#axiomsClean vector_index
#axiomsClean last_stride_layout
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
