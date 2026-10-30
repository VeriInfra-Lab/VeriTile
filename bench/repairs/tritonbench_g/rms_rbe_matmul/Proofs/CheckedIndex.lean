import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.CheckedNative

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
open VeriTile.Bench.Repairs.RmsRbeMatmul.Contract (Inputs InputContract)
open VeriTile.Bench.Repairs.RmsMatmulRbe.Checked (ActiveBounds pointer64 pointer_exact flat3 flat3_lt)
noncomputable section
set_option maxHeartbeats 5000000

/-- Quotient/remainder form used by the retained raw 2×1024 rotary helper. -/
theorem raw_safe_ceil (n : Nat) : RmsMatmulRbe.Checked.safeCeil n 1024=(n+1023)/1024 := by
  unfold RmsMatmulRbe.Checked.safeCeil
  split_ifs <;> omega

theorem active_bounds (i : Inputs) (h : InputContract i) (a : Allocation i)
    (active : 0<(shape i).B ∧ 0<(shape i).M) : ActiveBounds (shape i) := by
  have f := facts i h
  let s := shape i
  have hK : 0<s.K := f.valid.1
  have hN : 0<s.N := by rw [f.valid.2.2.2.2]; exact Nat.mul_pos f.valid.2.1 f.valid.2.2.1
  have hB : 0<s.B := active.1
  have hM : 0<s.M := active.2
  have hx : s.B*s.M*s.K<2^63 := by
    rw [←f.xCount]; exact (a.ordinary i.x (by simp [Contract.Inputs.reads])).count64
  have ho : s.B*s.M*s.N<2^63 := by rw [←a.outCount]; exact a.outOrdinary.count64
  have hw : s.N*s.K<2^63 := by
    rw [←f.wCount]; exact (a.ordinary i.weight (by simp [Contract.Inputs.reads])).count64
  have hMK : s.M*s.K<2^63 := by nlinarith [Nat.mul_pos hM hK]
  have hMN : s.M*s.N<2^63 := by nlinarith [Nat.mul_pos hM hN]
  have hBM : s.B*s.M≤s.B*s.M*s.K := by nlinarith [Nat.mul_pos hB hM]
  have hb : s.B<2^63 := by nlinarith
  have hm : s.M<2^63 := by nlinarith
  have hn : s.N<2^63 := by nlinarith
  have hk : s.K<2^63 := by nlinarith
  have tm : RmsMatmulRbe.tilesM s≤s.M := by unfold RmsMatmulRbe.tilesM; omega
  have tn : RmsMatmulRbe.tilesN s≤s.N := by unfold RmsMatmulRbe.tilesN; omega
  change ActiveBounds s
  refine ⟨⟨hb,hm,hn,hk⟩,hMK,hMN,(Nat.mul_le_mul tm tn).trans_lt hMN,?_,?_,?_⟩
  · unfold RmsMatmulRbe.tilesM; omega
  · unfold RmsMatmulRbe.tilesN; omega
  · unfold RmsMatmulRbe.steps; omega

theorem padded_position (i : Inputs) (h : InputContract i)
    (rotary : i.rotary=Bool.true) (active : 0<(shape i).B ∧ 0<(shape i).M)
    (row : Nat) (hr : row<tilesM (shape i)*16) :
    -(2^63:Int) ≤ i.scalars.start+(row:Int) ∧ i.scalars.start+(row:Int)<(2^63:Int) := by
  have f := facts i h
  have low := f.scalars.2.2.2.2.2.2.1
  have high := f.scalars.2.2.2.2.2.2.2.2.1 rotary active.1 active.2
  change -(2^63:Int) ≤ i.scalars.start at low
  change i.scalars.start+(16*(((shape i).M+15)/16):Nat)-1<(2^63:Int) at high
  unfold tilesM at hr
  omega

theorem output_address64 (i : Inputs) (a : Allocation i) (x : Cell (shape i)) :
    (pointer64 a.out (flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2)).toNat=
      a.out.base+RmsMatmulRbe.Contract.width a.out.metadata.dtype*(outputAddress (buffers i a) x).2 := by
  have hj : flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2<a.out.metadata.numel := by
    rw [a.outCount]; exact flat3_lt _ _ _ _ _ _ x.1.isLt x.2.1.isLt x.2.2.isLt
  rw [(pointer_exact a.out a.outOrdinary _ hj).1]
  change _=a.out.base+_*((dense3 a.out _ _).address _ _ _).2
  rw [dense3_address]
theorem output_pointer_valid (i : Inputs) (a : Allocation i) (x : Cell (shape i)) :
    RmsMatmulRbe.Checked.PointerValid a.out (flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2) := by
  apply RmsMatmulRbe.Checked.pointer_valid a.out a.outOrdinary a.outAdmitted
  rw [a.outCount]
  exact flat3_lt _ _ _ _ _ _ x.1.isLt x.2.1.isLt x.2.2.isLt

def LoadsValid (i : Inputs) (a : Allocation i) : Prop :=
  (∀ b m k,b<(shape i).B → m<(shape i).M → k<(shape i).K →
    RmsMatmulRbe.Checked.PointerValid i.x (flat3 (shape i).M (shape i).K b m k)) ∧
  (∀ n k,n<(shape i).N → k<(shape i).K → RmsMatmulRbe.Checked.PointerValid i.weight (n*(shape i).K+k)) ∧
  (∀ k,k<(shape i).K → RmsMatmulRbe.Checked.PointerValid a.rms k)
theorem loads_valid (i : Inputs) (h : InputContract i) (a : Allocation i) : LoadsValid i a := by
  have f := facts i h
  refine ⟨?_,?_,?_⟩
  · intro b m k hb hm hk
    apply RmsMatmulRbe.Checked.pointer_valid i.x (a.ordinary i.x (by simp [Contract.Inputs.reads])) f.xAdmitted
    rw [f.xCount]
    exact flat3_lt _ _ _ _ _ _ hb hm hk
  · intro n k hn hk
    apply RmsMatmulRbe.Checked.pointer_valid i.weight (a.ordinary i.weight (by simp [Contract.Inputs.reads])) f.weightAdmitted
    rw [f.wCount]
    nlinarith
  · intro k hk
    apply RmsMatmulRbe.Checked.pointer_valid a.rms a.rmsOrdinary a.rmsAdmitted
    simpa only [a.rmsCount] using hk

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
