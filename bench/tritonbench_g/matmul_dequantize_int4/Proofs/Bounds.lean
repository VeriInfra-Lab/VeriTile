import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Setup
import bench.tritonbench_g.matmul_kernel.Proofs.Bounds
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Bounds
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram)
set_option maxHeartbeats 4000000
/-- Source-specific signed-i32 envelope for all 17 candidates, including
masked row pointer formation and the last post-loop pointer increments.
The generic grid-product/launch-runtime project remains outside this package. -/
structure Signed32 (c : Call) : Prop where
  dims : c.M+128<2^31 ∧ 8*c.N+256<2^31 ∧ c.K+128<2^31 ∧ c.group<2^31
  strides : ∀ v∈[c.A,c.Q,c.S,c.Z,c.C],v.sr<2^31 ∧ v.sc<2^31
  aformed : (c.M+128)*c.A.sr+(c.K+128)*c.A.sc<2^31
  qformed : (c.K/8+16)*c.Q.sr+c.N*c.Q.sc<2^31
  sformed : (c.K/c.group)*c.S.sr+c.N*c.S.sc<2^31
  zformed : (c.K/c.group)*c.Z.sr+(c.N/8)*c.Z.sc<2^31
  cformed : (c.M+128)*c.C.sr+c.N*c.C.sc<2^31

theorem columns_full (c : Call) (h : c.Valid) (i : Fin 17) :
    blocksN (params c i)*(tunings i).BN=c.N := by
  have hd := dvd_trans (tuning_facts i).2.2.2.1 h.n256
  change ((c.N+(tunings i).BN-1)/(tunings i).BN)*(tunings i).BN=c.N
  rw [ceil_exact _ _ (tuning_facts i).2.2.1 hd]
  exact Nat.div_mul_cancel hd

theorem source_coordinates (c : Call) (h : c.Valid) (i : Fin 17) (s : State Word)
    (pid : Nat) (hp : pid<Geometry.count (Grid.geometry (params c i))) (lane : CIndex (params c i)) :
    inputRow (params c i) (atProgram s pid) lane.1<c.M+128 ∧
    inputColumn (params c i) (atProgram s pid) lane.2.1<c.N := by
  let p := params c i
  have hn : 0<blocksN p := by
    have he := columns_full c h i
    change blocksN p*(tunings i).BN=c.N at he
    have := h.n
    by_contra hh
    have hz : blocksN p=0 := by omega
    rw [hz,Nat.zero_mul] at he
    omega
  have hb := GroupedGrid.bounds (blocksM p) (blocksN p) 8 pid hn (by decide) hp
  have hm : inputRow p (atProgram s pid) lane.1<blocksM p*p.blockRows :=
    MatmulKernel.tiled_bound _ _ _ _ hb.1 lane.1.isLt
  have hc : inputColumn p (atProgram s pid) lane.2.1<blocksN p*p.blockColumns :=
    MatmulKernel.tiled_bound _ _ _ _ hb.2 lane.2.1.isLt
  refine ⟨?_,?_⟩
  · have hd := Nat.div_mul_le_self (c.M+(tunings i).BM-1) (tunings i).BM
    have ht := (tuning_facts i).2.1
    change inputRow p (atProgram s pid) lane.1<c.M+128
    change inputRow p (atProgram s pid) lane.1<((c.M+(tunings i).BM-1)/(tunings i).BM)*(tunings i).BM at hm
    omega
  · change inputColumn p (atProgram s pid) lane.2.1<blocksN (params c i)*(tunings i).BN at hc
    rw [columns_full c h i] at hc
    exact hc

theorem group_bound (c : Call) (h : c.Valid) (k : Nat) (hk : k<c.K) : k/c.group<c.K/c.group := by
  apply (Nat.div_lt_iff_lt_mul h.g).mpr
  simpa [Nat.div_mul_cancel h.groups] using hk

theorem allocation_bounds (c : Call) (h : c.Valid) (r k n : Nat)
    (hr : r<c.M) (hk : k<c.K) (hn : n<c.N) :
    c.A.addr r k<c.A.capacity ∧ c.Q.addr (k/8) n<c.Q.capacity ∧
    c.S.addr (k/c.group) n<c.S.capacity ∧ c.Z.addr (k/c.group) (n/8)<c.Z.capacity ∧
    c.C.addr r n<c.C.capacity := by
  have hk8 : k/8<c.K/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    simpa [Nat.div_mul_cancel (dvd_trans (by decide : 8∣128) h.k128)] using hk
  have hn8 : n/8<c.N/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    simpa [Nat.div_mul_cancel (dvd_trans (by decide : 8∣256) h.n256)] using hn
  have hg := group_bound c h k hk
  refine ⟨MatmulKernel.view_bound c.A h.aValid r k ?_ ?_,
    MatmulKernel.view_bound c.Q h.qValid (k/8) n ?_ ?_,
    MatmulKernel.view_bound c.S h.sValid (k/c.group) n ?_ ?_,
    MatmulKernel.view_bound c.Z h.zValid (k/c.group) (n/8) ?_ ?_,
    MatmulKernel.view_bound c.C h.cValid r n ?_ ?_⟩
  all_goals simp only [h.aShape.1,h.aShape.2,h.qShape.1,h.qShape.2,h.sShape.1,h.sShape.2,
    h.zShape.1,h.zShape.2,h.cShape.1,h.cShape.2]
  all_goals assumption

theorem formed_k (c : Call) (h : c.Valid) (i : Fin 17) (step t : Nat)
    (hs : step≤steps (params c i)) (ht : t<(tunings i).BK) :
    step*(tunings i).BK+t<c.K+128 := by
  have hh := Nat.mul_le_mul_right (tunings i).BK hs
  have he := (legal c h i).fullK
  change steps (params c i)*(tunings i).BK=c.K at he
  rw [he] at hh
  have hb : (tunings i).BK≤128 := Nat.le_of_dvd (by decide) (tuning_facts i).2.2.2.2.2.1
  omega

theorem formed_offsets (c : Call) (hs : Signed32 c) (r k n g : Nat)
    (hr : r<c.M+128) (hk : k<c.K+128) (hn : n<c.N)
    (hg : g<c.K/c.group) (hk8 : k/8<c.K/8+16) :
    c.A.addr r k<2^31 ∧ c.Q.addr (k/8) n<2^31 ∧ c.S.addr g n<2^31 ∧
    c.Z.addr g (n/8)<2^31 ∧ c.C.addr r n<2^31 := by
  have a1 := Nat.mul_le_mul_right c.A.sr (Nat.le_of_lt hr)
  have a2 := Nat.mul_le_mul_right c.A.sc (Nat.le_of_lt hk)
  have q1 := Nat.mul_le_mul_right c.Q.sr (Nat.le_of_lt hk8)
  have q2 := Nat.mul_le_mul_right c.Q.sc (Nat.le_of_lt hn)
  have s1 := Nat.mul_le_mul_right c.S.sr (Nat.le_of_lt hg)
  have s2 := Nat.mul_le_mul_right c.S.sc (Nat.le_of_lt hn)
  have z1 := Nat.mul_le_mul_right c.Z.sr (Nat.le_of_lt hg)
  have z2 := Nat.mul_le_mul_right c.Z.sc (Nat.div_le_div_right (Nat.le_of_lt hn) : n/8≤c.N/8)
  have c1 := Nat.mul_le_mul_right c.C.sr (Nat.le_of_lt hr)
  have c2 := Nat.mul_le_mul_right c.C.sc (Nat.le_of_lt hn)
  have := hs.aformed
  have := hs.qformed
  have := hs.sformed
  have := hs.zformed
  have := hs.cformed
  unfold MatmulKernel.View.addr
  omega

theorem formed_packed_k (c : Call) (h : c.Valid) (k : Nat) (hk : k<c.K+128) :
    k/8<c.K/8+16 := by
  apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
  have he := Nat.div_mul_cancel (dvd_trans (by decide : 8∣128) h.k128)
  rw [Nat.add_mul,he]
  exact hk

theorem source_qsz_loads (c : Call) (h : c.Valid) (i : Fin 17) (s : State Word) (pid step : Nat)
    (hp : pid<Geometry.count (Grid.geometry (params c i))) (hs : step<steps (params c i))
    (lane : BIndex (params c i)) :
    bOffset (params c i) (atProgram s pid) step lane<c.Q.capacity ∧
    scaleOffset (params c i) (atProgram s pid) step (lane.2.1,PUnit.unit)<c.S.capacity ∧
    zeroOffset (params c i) (atProgram s pid) step (lane.2.1,PUnit.unit)<c.Z.capacity := by
  let p := params c i
  let st := atProgram s pid
  let k := step*p.blockInner+lane.1.val
  let n := inputColumn p st lane.2.1
  have hk : k<c.K := Logical.k_bound p (legal c h i) step hs lane.1
  have hn : n<c.N := (source_coordinates c h i s pid hp
    (⟨0,(tuning_facts i).1⟩,lane.2.1,PUnit.unit)).2
  have all := allocation_bounds c h 0 k n h.m hk hn
  have hd := Logical.packed_div p (legal c h i) step lane.1.val
  have hg := Logical.group_eq p (legal c h i) step hs lane.1
  have eqQ : bOffset p st step lane=c.Q.addr (k/8) n := by
    simp only [bOffset,inputInner,MatmulKernel.View.addr,k,hd]
    change lane.1.val/8*c.Q.sr+n*c.Q.sc+step*(p.blockInner/8*c.Q.sr)=
      (step*(p.blockInner/8)+lane.1.val/8)*c.Q.sr+n*c.Q.sc
    ring
  have eqS : scaleOffset p st step (lane.2.1,PUnit.unit)=c.S.addr (k/c.group) n := by
    simp only [scaleOffset,hg,MatmulKernel.View.addr]
    change n*c.S.sc+k/c.group*c.S.sr=k/c.group*c.S.sr+n*c.S.sc
    omega
  have eqZ : zeroOffset p st step (lane.2.1,PUnit.unit)=c.Z.addr (k/c.group) (n/8) := by
    simp only [zeroOffset,hg,MatmulKernel.View.addr]
    change n/8*c.Z.sc+k/c.group*c.Z.sr=k/c.group*c.Z.sr+n/8*c.Z.sc
    omega
  change bOffset p st step lane<_ ∧ scaleOffset p st step (lane.2.1,PUnit.unit)<_ ∧
    zeroOffset p st step (lane.2.1,PUnit.unit)<_
  rw [eqQ,eqS,eqZ]
  exact ⟨all.2.1,all.2.2.1,all.2.2.2.1⟩

theorem source_a_load (c : Call) (h : c.Valid) (i : Fin 17) (s : State Word) (step : Nat)
    (hs : step<steps (params c i)) (lane : AIndex (params c i))
    (hm : inputRow (params c i) s lane.1<c.M) :
    aOffset (params c i) s step lane<c.A.capacity := by
  have hk := Logical.k_bound (params c i) (legal c h i) step hs lane.2.1
  have hv := MatmulKernel.view_bound c.A h.aValid (inputRow (params c i) s lane.1)
    (step*(tunings i).BK+lane.2.1.val) (by simpa only [h.aShape.1] using hm)
    (by simpa only [h.aShape.2] using hk)
  have eq : aOffset (params c i) s step lane=c.A.addr (inputRow (params c i) s lane.1)
      (step*(tunings i).BK+lane.2.1.val) := by
    simp only [aOffset,inputInner,MatmulKernel.View.addr,params]
    ring
  rw [eq]
  exact hv

theorem source_c_store (c : Call) (h : c.Valid) (i : Fin 17) (s : State Word)
    (lane : CIndex (params c i)) (hm : active (params c i) s lane) :
    outputOffset (params c i) s lane<c.C.capacity := by
  have hv := MatmulKernel.view_bound c.C h.cValid (inputRow (params c i) s lane.1)
    (inputColumn (params c i) s lane.2.1)
    (by simpa only [h.cShape.1] using hm.1) (by simpa only [h.cShape.2] using hm.2)
  convert hv using 1 <;> simp only [outputOffset,inputRow,inputColumn,MatmulKernel.View.addr,params,Nat.mul_comm]

theorem shift_bound (i : Nat) : (i%8)*4≤28 := by have := Nat.mod_lt i (by decide : 0<8); omega
#axiomsClean columns_full
#axiomsClean source_coordinates
#axiomsClean allocation_bounds
#axiomsClean formed_k
#axiomsClean formed_offsets
#axiomsClean source_qsz_loads
#axiomsClean source_a_load
#axiomsClean formed_packed_k
#axiomsClean source_c_store
#axiomsClean shift_bound
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Bounds
