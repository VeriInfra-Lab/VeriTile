import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedMemory

namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
noncomputable section
set_option maxHeartbeats 5000000
open Contract

def safeCeil (n d : Nat) := n/d+if n%d=0 then 0 else 1
theorem safeCeil16 (n : Nat) : safeCeil n 16=(n+15)/16 := by
  unfold safeCeil; split_ifs <;> omega
theorem safeCeil64 (n : Nat) : safeCeil n 64=(n+63)/64 := by
  unfold safeCeil; split_ifs <;> omega
theorem loop_mask64 (K step lane : Nat) (hstep : step<steps K) (hlane : lane<64) :
    step*64<K ∧ (lane<K-step*64 ↔ step*64+lane<K) := by
  unfold steps at hstep
  omega

structure ActiveBounds (s : Shape) : Prop where
  dimensions : s.B<2^63 ∧ s.M<2^63 ∧ s.N<2^63 ∧ s.K<2^63
  xBatchStride : s.M*s.K<2^63
  outBatchStride : s.M*s.N<2^63
  grid : tilesM s*tilesN s<2^63
  paddedM : tilesM s*16≤2^63
  paddedN : tilesN s*64≤2^63
  paddedK : steps s.K*64≤2^63

theorem active_bounds (i : Inputs) (h : InputContract i) (a : Allocation i)
    (active : 0<(shape i).B ∧ 0<(shape i).M) : ActiveBounds (shape i) := by
  have f := facts i h
  let s := shape i
  have hK : 0<s.K := f.valid.1
  have hN : 0<s.N := by rw [f.valid.2.2.2.2]; exact Nat.mul_pos f.valid.2.1 f.valid.2.2.1
  have hB : 0<s.B := active.1
  have hM : 0<s.M := active.2
  have hx : s.B*s.M*s.K<2^63 := by
    rw [←f.xCount]; exact (a.ordinary i.x (by simp [Inputs.reads])).count64
  have ho : s.B*s.M*s.N<2^63 := by rw [←a.qCount]; exact a.qOrdinary.count64
  have hw : s.N*s.K<2^63 := by
    rw [←f.wCount 0]; exact (a.ordinary (weight i 0) (by simp [weight_mem])).count64
  have hMK : s.M*s.K<2^63 := by nlinarith [Nat.mul_pos hM hK]
  have hMN : s.M*s.N<2^63 := by nlinarith [Nat.mul_pos hM hN]
  have hBM : s.B*s.M≤s.B*s.M*s.K := by nlinarith [Nat.mul_pos hB hM]
  have hb : s.B<2^63 := by nlinarith
  have hm : s.M<2^63 := by nlinarith
  have hn : s.N<2^63 := by nlinarith
  have hk : s.K<2^63 := by nlinarith
  have tm : tilesM s≤s.M := by unfold tilesM; omega
  have tn : tilesN s≤s.N := by unfold tilesN; omega
  change ActiveBounds s
  refine ⟨⟨hb,hm,hn,hk⟩,hMK,hMN,(Nat.mul_le_mul tm tn).trans_lt hMN,?_,?_,?_⟩
  · unfold tilesM; omega
  · unfold tilesN; omega
  · unfold steps; omega

theorem padded_position (i : Inputs) (h : InputContract i)
    (active : 0<(shape i).B ∧ 0<(shape i).M) (row : Nat) (hr : row<tilesM (shape i)*16) :
    -(2^63:Int) ≤ i.scalars.start+(row:Int) ∧ i.scalars.start+(row:Int)<(2^63:Int) := by
  have f := facts i h
  have low := f.scalars.2.2.2.2.2.2.1
  have high := f.scalars.2.2.2.2.2.2.2.2.1 rfl active.1 active.2
  unfold tilesM at hr
  omega

theorem output_address64 (i : Inputs) (h : InputContract i) (a : Allocation i) (x : Cell (shape i)) :
    (pointer64 (output i a.q x.1) (flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2)).toNat=
      (output i a.q x.1).base+width (output i a.q x.1).metadata.dtype*
        (outputAddress (buffers i a.q) x).2 := by
  have hj : flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2<(output i a.q x.1).metadata.numel := by
    rw [output_count i h a]
    exact flat3_lt _ _ _ _ _ _ x.2.1.isLt x.2.2.1.isLt x.2.2.2.isLt
  rw [(pointer_exact _ (output_ordinary a x.1) _ hj).1]
  change _=(output i a.q x.1).base+_*
    ((dense3 (output i a.q x.1) _ _).address _ _ _).2
  rw [dense3_address]
theorem output_pointer_valid (i : Inputs) (h : InputContract i) (a : Allocation i) (x : Cell (shape i)) :
    PointerValid (output i a.q x.1) (flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2) := by
  apply pointer_valid _ (output_ordinary a x.1) (output_admitted i h a x.1)
  rw [output_count i h a]
  exact flat3_lt _ _ _ _ _ _ x.2.1.isLt x.2.2.1.isLt x.2.2.2.isLt

def LoadsValid (i : Inputs) : Prop :=
  (∀ b m k,b<(shape i).B → m<(shape i).M → k<(shape i).K →
    PointerValid i.x (flat3 (shape i).M (shape i).K b m k)) ∧
  (∀ phase n k,n<(shape i).N → k<(shape i).K → PointerValid (weight i phase) (n*(shape i).K+k)) ∧
  (∀ k,k<(shape i).K → PointerValid i.rms k)
theorem loads_valid (i : Inputs) (h : InputContract i) (a : Allocation i) : LoadsValid i := by
  have f := facts i h
  refine ⟨?_,?_,?_⟩
  · intro b m k hb hm hk
    apply pointer_valid i.x (a.ordinary i.x (by simp [Inputs.reads])) f.xAdmitted
    rw [f.xCount]
    exact flat3_lt _ _ _ _ _ _ hb hm hk
  · intro phase n k hn hk
    have ho : weight i phase∈i.others := by fin_cases phase <;> simp [weight,Inputs.others]
    apply pointer_valid (weight i phase) (a.ordinary _ (by simp [weight_mem])) (f.others _ ho)
    rw [f.wCount]
    nlinarith
  · intro k hk
    apply pointer_valid i.rms (a.ordinary i.rms (by simp [Inputs.reads])) (f.others i.rms (by simp [Inputs.others]))
    simpa only [f.rCount] using hk

theorem wrapped_loads_valid (s : Shape) (valid : s.Valid) (active : 0<s.M)
    (tileM tileN row col step lane : Nat) (hs : step<steps s.K) (hl : lane<64)
    (mask : lane<s.K-step*64) :
    (tileM*16+row)%s.M<s.M ∧ (tileN*64+col)%s.N<s.N ∧ step*64+lane<s.K := by
  have hN : 0<s.N := by rw [valid.2.2.2.2]; exact Nat.mul_pos valid.2.1 valid.2.2.1
  exact ⟨Nat.mod_lt _ active,Nat.mod_lt _ hN,(loop_mask64 s.K step lane hs hl).2.mp mask⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
