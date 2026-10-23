import bench.repairs.tritonbench_g.lightning_attention.Proofs.InterModel
import bench.repairs.tritonbench_g.chunk_delta_fwd.Readback
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem row_feature_collision (width row other feature col : Nat)
    (hf : feature<width) (hc : col<width)
    (he : row*width+feature=other*width+col) : row=other ∧ feature=col := by
  have hm := congrArg (fun x => x%width) he
  have hfc : feature=col := by
    simpa [Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt hf,Nat.mod_eq_of_lt hc] using hm
  refine ⟨Nat.eq_of_mul_eq_mul_right (m:=width) (n:=row) (k:=other) (by omega) (by omega),hfc⟩

theorem writeRows_read (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (m : Memory α) (i : TileIndex [rows,width])
    (hi : start+i.1.val<N) :
    writeRows M r N width head start rows values m r (head*N*width+(start+i.1.val)*width+i.2.1.val)=
      .mk .real (cast M (values i)) := by
  apply ChunkDeltaFwd.scatter_readback _ _ _ _ _ i hi
  intro a b ha hb he
  have hh : (start+a.1.val)*width+a.2.1.val=(start+b.1.val)*width+b.2.1.val := by omega
  have h := row_feature_collision width _ _ _ _ a.2.1.isLt b.2.1.isLt hh
  apply Prod.ext
  · apply Fin.ext;omega
  · apply Prod.ext (Fin.ext h.2);rfl

theorem writeTrans_read (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (m : Memory α) (i : TileIndex [width,rows])
    (hi : start+i.2.1.val<N) :
    writeTrans M r N width head start rows values m r (head*N*width+(start+i.2.1.val)*width+i.1.val)=
      .mk .real (cast M (values i)) := by
  apply ChunkDeltaFwd.scatter_readback _ _ _ _ _ i hi
  intro a b ha hb he
  have hh : (start+a.2.1.val)*width+a.1.val=(start+b.2.1.val)*width+b.1.val := by omega
  have h := row_feature_collision width _ _ _ _ a.1.isLt b.1.isLt hh
  apply Prod.ext (Fin.ext h.2)
  apply Prod.ext
  · apply Fin.ext;omega
  · rfl

theorem writeRows_frame (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (m : Memory α) (region : RegionName) (address : Nat)
    (safe : region≠r ∨ ∀ i : TileIndex [rows,width], start+i.1.val<N →
      address≠head*N*width+(start+i.1.val)*width+i.2.1.val) :
    writeRows M r N width head start rows values m region address=m region address := by
  apply scatter_frame
  rcases safe with hr|h
  · exact Or.inl hr
  · exact Or.inr (fun i _ => h i)

theorem writeTrans_frame (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (m : Memory α) (region : RegionName) (address : Nat)
    (safe : region≠r ∨ ∀ i : TileIndex [width,rows], start+i.2.1.val<N →
      address≠head*N*width+(start+i.2.1.val)*width+i.1.val) :
    writeTrans M r N width head start rows values m region address=m region address := by
  apply scatter_frame
  rcases safe with hr|h
  · exact Or.inl hr
  · exact Or.inr (fun i _ => h i)

theorem writeRows_time_frame (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (m : Memory α) (time : Nat) (feature : Fin width)
    (safe : ∀ lane : Fin rows,start+lane.val≠time) :
    writeRows M r N width head start rows values m r (head*N*width+time*width+feature.val)=
      m r (head*N*width+time*width+feature.val) := by
  apply writeRows_frame
  right
  intro i hi he
  have h := row_feature_collision width time (start+i.1.val) feature.val i.2.1.val feature.isLt i.2.1.isLt
    (show time*width+feature.val=(start+i.1.val)*width+i.2.1.val by omega)
  exact safe i.1 h.1.symm

theorem writeTrans_time_frame (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (m : Memory α) (time : Nat) (feature : Fin width)
    (safe : ∀ lane : Fin rows,start+lane.val≠time) :
    writeTrans M r N width head start rows values m r (head*N*width+time*width+feature.val)=
      m r (head*N*width+time*width+feature.val) := by
  apply writeTrans_frame
  right
  intro i hi he
  have h := row_feature_collision width time (start+i.2.1.val) feature.val i.1.val feature.isLt i.1.isLt
    (show time*width+feature.val=(start+i.2.1.val)*width+i.1.val by omega)
  exact safe i.2.1 h.1.symm

theorem queryUpdate_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.E p.D) (m : Memory α) :
    Inputs (queryUpdate M D p head row carry m) m :=
  writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _

theorem keyValueUpdate_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.D p.E) (m : Memory α) :
    Inputs (keyValueUpdate M D p head row carry m) m :=
  (writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _).trans
    (writeTrans_inputs _ _ (by decide) _ _ _ _ _ _ _)

theorem queryUpdate_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.E p.D) (m : Memory α)
    (i : TileIndex [p.CB,p.D]) (ht : row+i.1.val<p.N) :
    queryUpdate M D p head row carry m "queryGradient" (head*p.N*p.D+(row+i.1.val)*p.D+i.2.1.val)=
      .mk .real (cast M (add M (dot D (subG M p head row m) carry i)
        ((m "queryGradient" (head*p.N*p.D+(row+i.1.val)*p.D+i.2.1.val)).read .real))) := by
  unfold queryUpdate
  rw [writeRows_read _ _ _ _ _ _ _ _ _ i ht]
  simp only [gradientRows,ht,ite_true]

theorem keyUpdate_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.D p.E) (m : Memory α)
    (i : TileIndex [p.D,p.CB]) (ht : row+i.2.1.val<p.N) :
    keyValueUpdate M D p head row carry m "keyGradient" (head*p.N*p.D+(row+i.2.1.val)*p.D+i.1.val)=
      .mk .real (cast M (add M (dot D carry (subVT M p head row m) i)
        ((m "keyGradient" (head*p.N*p.D+(row+i.2.1.val)*p.D+i.1.val)).read .real))) := by
  unfold keyValueUpdate
  rw [writeRows_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
  rw [writeTrans_read _ _ _ _ _ _ _ _ _ i ht]
  simp only [gradientTrans,ht,ite_true]

theorem valueUpdate_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.D p.E) (m : Memory α)
    (i : TileIndex [p.CB,p.E]) (ht : row+i.1.val<p.N) :
    keyValueUpdate M D p head row carry m "valueGradient" (head*p.N*p.E+(row+i.1.val)*p.E+i.2.1.val)=
      .mk .real (cast M (add M (dot D (subK M p head row m) carry i)
        ((m "valueGradient" (head*p.N*p.E+(row+i.1.val)*p.E+i.2.1.val)).read .real))) := by
  unfold keyValueUpdate
  rw [writeRows_read _ _ _ _ _ _ _ _ _ i ht]
  simp only [gradientRows,ht,ite_true]

#axiomsClean writeRows_read
#axiomsClean writeTrans_read
#axiomsClean queryUpdate_read
#axiomsClean keyUpdate_read
#axiomsClean valueUpdate_read
end
end VeriTile.Bench.Repairs.LightningAttention
