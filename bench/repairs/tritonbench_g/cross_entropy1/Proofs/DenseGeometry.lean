import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardKernel

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

theorem matrix_row_start (t : Tensor) (rows cols row : Nat)
    (shape : t.shape=[rows,cols]) (rank : t.strides.length=t.shape.length)
    (hr : row<rows) (positive : 0<cols) : t.index (row*cols)=t.storageOffset+row*rowStride t := by
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
        simpa [rowStride] using matrix_index ⟨objectId,region,shape',[rs,cs],offset,neg,cuda,device,dtype⟩
          rows cols rs cs row 0 shape rfl hr positive

theorem vector_layout (t : Tensor) (rows row : Nat) (shape : t.shape=[rows])
    (rank : t.strides.length=t.shape.length) (hr : row<rows) :
    t.index row=t.storageOffset+row*rowStride t := by
  have len : t.strides.length=1 := by simpa [shape] using rank
  rcases t with ⟨objectId,region,shape',strides,offset,neg,cuda,device,dtype⟩
  dsimp at *
  cases strides with
  | nil => simp at len
  | cons stride tail =>
    cases tail with
    | cons a rest => simp at len
    | nil =>
      simpa [rowStride] using vector_index ⟨objectId,region,shape',[stride],offset,neg,cuda,device,dtype⟩
        rows stride row shape rfl hr

theorem dense_rows (t : Tensor) (rows cols : Nat) (dense : t.Dense)
    (shape : t.shape=[rows,cols]) (rank : t.strides.length=t.shape.length) :
    rows≤1 ∨ cols≤rowStride t := by
  by_cases single : rows≤1
  · exact Or.inl single
  · right
    by_cases empty : cols=0
    · simp [empty]
    · have size : t.numel=rows*cols := by simp [Tensor.numel,shape]
      have bound : cols<t.numel := by
        rw [size]
        have product := Nat.mul_le_mul_right cols (show 2≤rows by omega)
        omega
      have base := dense_base t dense (by omega)
      have index := matrix_row_start t rows cols 1 shape rank (by omega) (by omega)
      have atOne := dense.2 cols bound
      simp only [Nat.one_mul,base,Nat.zero_add,atOne] at index
      exact Nat.le_of_eq index

theorem dense_row_layout (t : Tensor) (dense : t.Dense) (rank : t.strides.length=t.shape.length) : RowLayout t := by
  intro rows cols shape row col hr hc
  have size : t.numel=rows*cols := by simp [Tensor.numel,shape]
  have bound : row*cols+col<t.numel := by
    rw [size]
    have product := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at product
    omega
  have base := dense_base t dense (by omega)
  rw [dense.2 _ bound,base,Nat.zero_add]
  by_cases zeroRow : row=0
  · simp [zeroRow]
  · have oneBound : cols<t.numel := by
      rw [size]
      have product := Nat.mul_le_mul_right cols (show 2≤rows by omega)
      omega
    have atOne := matrix_row_start t rows cols 1 shape rank (by omega) (by omega)
    simp only [Nat.one_mul,base,Nat.zero_add,dense.2 cols oneBound] at atOne
    rw [← atOne]

#axiomsClean dense_rows
#axiomsClean dense_row_layout
#axiomsClean vector_layout
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
