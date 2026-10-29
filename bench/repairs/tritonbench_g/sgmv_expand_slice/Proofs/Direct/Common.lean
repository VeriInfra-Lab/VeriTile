import bench.repairs.tritonbench_g.sgmv_expand_slice.Contract
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedMemory

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Direct
open Contract
noncomputable section
open Classical

/-- Canonical strides passed to the kernel for the original contiguous views.
The historical PreparedMemory region strings are logical role names only. -/
def strides (p : Shape) : PreparedMemory.Strides :=
  ⟨p.inner,1,p.columns*p.inner,p.inner,1,p.outputColumns,1⟩
def args (t : Inputs) (s : Scalars) := PreparedMemory.arguments (shape t s) (strides (shape t s))

theorem xlayout (p : Shape) :
    GridStride.Contiguous p.tokens p.inner (strides p).inputRow (strides p).inputInner :=
  ⟨fun _ _ => rfl,fun _ _ => rfl⟩
theorem wlayout (p : Shape) :
    SnapshotLayout.WeightContiguous p.adapters p.columns p.inner
      (strides p).weightAdapter (strides p).weightColumn (strides p).weightInner :=
  ⟨fun _ _ _ => rfl,fun _ _ _ => rfl,fun _ _ _ => rfl⟩
theorem ylayout (p : Shape) :
    GridStride.Contiguous p.outputRows p.outputColumns (strides p).outputRow (strides p).outputColumn :=
  ⟨fun _ _ => rfl,fun _ _ => rfl⟩

theorem metadata_projections (ok : Admitted t s lists) :
    (raw t s lists).map RawSegment.start=lists.starts ∧
    (raw t s lists).map RawSegment.length++lists.lengths.drop s.batches.toNat=lists.lengths ∧
    (raw t s lists).map RawSegment.adapter=lists.indices := by
  have c:=(python_checks_iff t s lists).mp ok
  have hs : lists.starts.length=s.batches.toNat := lists.startCount.trans c.shapes.2.2.2.1
  have hl : s.batches.toNat≤lists.lengths.length := by rw [lists.lengthCount];exact c.shapes.2.2.2.2.2
  have hi : lists.indices.length=s.batches.toNat := lists.indexCount.trans c.shapes.2.2.2.2.1
  have p:=HostMetadata.projections s.batches.toNat lists.starts lists.lengths lists.indices hs hl hi
  exact ⟨p.1,HostMetadata.full_lengths _ _ _ _ hs hl hi,p.2.2⟩

theorem matrix_numel (t : TensorInfo) (two : t.shape.length=2) : t.numel=rows t*columns t := by
  obtain ⟨a,b,h⟩:=List.length_eq_two.mp two
  simp [VeriTile.Torch.TensorMetadata.numel,rows,columns,
    VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
    VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns,h]
theorem input_numel (ok : Admitted t s lists) : t.x.numel=(shape t s).tokens*(shape t s).inner := by
  have c:=(python_checks_iff t s lists).mp ok
  simp [VeriTile.Torch.TensorMetadata.numel,c.shapes.1,shape]
theorem output_numel (ok : Admitted t s lists) : t.y.numel=(shape t s).outputRows*(shape t s).outputColumns :=
  matrix_numel t.y ((python_checks_iff t s lists).mp ok).ranks.2.1
theorem weight_numel (ok : Admitted t s lists) :
    t.w.numel=(shape t s).adapters*((shape t s).columns*(shape t s).inner) := by
  have c:=(python_checks_iff t s lists).mp ok
  rcases c.ranks.2.2.1 with three|four
  · obtain ⟨a,b,d,h⟩:=List.length_eq_three.mp three
    simp [VeriTile.Torch.TensorMetadata.numel,shape,rows,width,rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,h]
  · obtain ⟨a,b,d,e,h⟩:=List.length_eq_four.mp four
    have singleton : b=1 := by simpa [h] using c.singleton four
    simp [VeriTile.Torch.TensorMetadata.numel,shape,rows,width,rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,h,singleton]

def Touched (p : Shape) (segments : List Segment) (cell : Nat×Nat) : Prop :=
  ∃ segment∈segments,ActiveCell p segment cell.1 cell.2
def cells (p : Shape) (segments : List Segment) : List (Nat×Nat) :=
  ((List.range p.outputRows).product (List.range p.outputColumns)).filter
    (fun cell => decide (Touched p segments cell))
theorem mem_cells (p : Shape) (segments : List Segment) (cell : Nat×Nat) :
    cell∈cells p segments ↔ cell.1<p.outputRows ∧ cell.2<p.outputColumns ∧ Touched p segments cell := by
  rcases cell with ⟨r,c⟩
  simp [cells,and_assoc]
theorem touched_inside (ok : Admitted t s lists) (cell : Nat×Nat)
    (live : Touched (shape t s) (segments t s lists) cell) :
    cell.1<(shape t s).outputRows ∧ cell.2<(shape t s).outputColumns := by
  rcases live with ⟨segment,member,active,rowLo,rowHi,colLo,colHi⟩
  have f:=(fitted ok segment member).output active
  have c:=capacity ok
  omega
theorem touched_complete (ok : Admitted t s lists) (cell : Nat×Nat)
    (live : Touched (shape t s) (segments t s lists) cell) :
    cell∈cells (shape t s) (segments t s lists) :=
  (mem_cells _ _ _).mpr ⟨(touched_inside ok cell live).1,(touched_inside ok cell live).2,live⟩
def offset (p : Shape) (cell : Nat×Nat) := cell.1*p.outputColumns+cell.2
theorem output_inside (ok : Admitted t s lists) (cell : Nat×Nat)
    (member : cell∈cells (shape t s) (segments t s lists)) : offset (shape t s) cell<t.y.numel := by
  have h:=(mem_cells _ _ _).mp member
  rw [output_numel ok]
  exact linear_lt h.1 h.2.1

#axiomsClean metadata_projections
#axiomsClean input_numel
#axiomsClean output_numel
#axiomsClean weight_numel
#axiomsClean touched_complete
#axiomsClean output_inside
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Direct
