import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Real
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton Contract Direct
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent K-term GEMV over original logical Tensor coordinates. The
Real reader is arbitrary; half/float casts are erased only in this projection. -/
def tensorTarget (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (read : Nat → ℝ) (p : Program r heap h) : ℝ :=
  Target.output (call r heap h).K
    (fun k => logical read r.input.tensor (p.batch*(call r heap h).K+k))
    (fun k => read (r.w.storageBase+2*r.weights.physicalIndex
      ((indices r heap p.batch).toNat*((call r heap h).N*(call r heap h).K)+p.row*(call r heap h).K+k)))
    (logical read r.output.tensor (outputIndex r heap h p)) r.scalars.add Bool.true

theorem tensor_formula (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (read : Nat → ℝ) (p : Program r heap h) (active : indices r heap p.batch≠-1) :
    Real.formula (call r heap h) p (Real.initial r heap read)=tensorTarget r heap h read p := by
  have c:=checked r heap h
  have selected:=((Contract.selected_valid r.x r.w r.y r.i r.scalars (indices r heap) h
    p.batch p.row p.batchInside p.rowInside).resolve_left active).2
  have inp (k : Fin (call r heap h).K) :
      (Real.initial r heap read).input (p.batch*(call r heap h).K+k.val)=
        logical read r.input.tensor (p.batch*(call r heap h).K+k.val) := by
    have bound:=Bounds.active r.x r.w r.y r.i r.scalars (indices r heap) h p _ selected k.val k.isLt
    exact (logical_direct r.input read c.dense.1.1 _ bound.1).symm
  have weight (k : Fin (call r heap h).K) :
      (Real.initial r heap read).weights
        ((indices r heap p.batch).toNat*((call r heap h).N*(call r heap h).K)+p.row*(call r heap h).K+k.val)=
      read (r.w.storageBase+2*r.weights.physicalIndex
        ((indices r heap p.batch).toNat*((call r heap h).N*(call r heap h).K)+p.row*(call r heap h).K+k.val)) := by
    have bound:=Bounds.active r.x r.w r.y r.i r.scalars (indices r heap) h p _ selected k.val k.isLt
    have bound' : (indices r heap p.batch).toNat*((call r heap h).N*(call r heap h).K)+p.row*(call r heap h).K+k.val<r.w.numel := by
      simpa only [Bounds.weightOffset,Nat.zero_mul,Nat.add_zero] using bound.2.1
    rw [raw_pointer r.weights c.dense.2.1.1 _ bound']
    rfl
  have old : (Real.initial r heap read).output (SourceGrid.address (call r heap h) p)=
      logical read r.output.tensor (outputIndex r heap h p) :=
    (logical_direct r.output read c.dense.2.2.1.1 _ (output_inside r heap h p)).symm
  unfold Real.formula tensorTarget Target.output Target.dot
  simp only [↓reduceIte]
  change (if r.scalars.add then _ else _)=_
  simp_rw [show (Real.initial r heap read).indices p.batch=indices r heap p.batch from rfl,inp,weight,old]

#axiomsClean tensor_formula
#specNonCircular tensorTarget avoiding [Real.execute,Real.formula,Signed.Grid.run,FP.Structural.exec,Surface.source]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
