import bench.tritonbench_g.matmul_dequantize_int4.Proofs.IOData

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOBounds
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4.FPTyped
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram)
noncomputable section

structure Windows (c : RawKernel.Layout) (bounds : RegionName → Nat) : Prop where
  a : ∀ r k, r<c.M → k<c.K → c.A.addr r k<bounds "A"
  q : ∀ k n, k<c.K/8 → n<c.N → c.Q.addr k n<bounds "Q"
  s : ∀ g n, g<c.K/c.group → n<c.N → c.S.addr g n<bounds "S"
  z : ∀ g n, g<c.K/c.group → n<c.N/8 → c.Z.addr g n<bounds "Z"
  c : ∀ r n, r<c.M → n<c.N → c.C.addr r n<bounds "C"

theorem windows {α : Type} (c : RawKernel.Layout) (jobs : List (Invocation α))
    (x : ChannelValues α (IOData.inputs c)) (bounds : RegionName → Nat)
    (h : (IOData.program α c jobs).InBounds bounds x) : Windows c bounds := by
  constructor
  · intro r k hr hk
    exact h.2.1 (⟨0,by decide⟩ : Fin 4) (⟨r,hr⟩,⟨k,hk⟩,()) trivial
  · intro k n hk hn
    exact h.2.1 (⟨1,by decide⟩ : Fin 4) (⟨k,hk⟩,⟨n,hn⟩,()) trivial
  · intro g n hg hn
    exact h.2.1 (⟨2,by decide⟩ : Fin 4) (⟨g,hg⟩,⟨n,hn⟩,()) trivial
  · intro g n hg hn
    exact h.2.1 (⟨3,by decide⟩ : Fin 4) (⟨g,hg⟩,⟨n,hn⟩,()) trivial
  · intro r n hr hn
    exact h.2.2.1 (⟨0,by decide⟩ : Fin 1) (⟨r,hr⟩,⟨n,hn⟩,()) trivial

theorem column_bound {α : Type} (c : RawKernel.Layout) (h : c.Valid)
    (i : Fin 17) (s : State α) (pid : Nat)
    (hp : pid<KernelSource.count c.call i) (j : Fin (tunings i).BN) :
    inputColumn (params c.call i) (atProgram s pid) j<c.N := by
  let p := params c.call i
  have hn : 0<blocksN p := by
    have he := Bounds.columns_full c.call (c.valid h) i
    have := h.n
    change blocksN p*(tunings i).BN=c.N at he
    by_contra hh
    have hz : blocksN p=0 := by omega
    rw [hz,Nat.zero_mul] at he
    omega
  have hb := GroupedGrid.bounds (blocksM p) (blocksN p) 8 pid hn (by decide) hp
  have hc : inputColumn p (atProgram s pid) j<blocksN p*p.blockColumns :=
    MatmulKernel.tiled_bound _ _ _ _ hb.2 j.isLt
  change inputColumn p (atProgram s pid) j<blocksN (params c.call i)*(tunings i).BN at hc
  rw [Bounds.columns_full c.call (c.valid h) i] at hc
  exact hc

theorem steps_positive (c : RawKernel.Layout) (h : c.Valid) (i : Fin 17) :
    0<steps (params c.call i) := by
  have he := (legal c.call (c.valid h) i).fullK
  have := h.k
  change steps (params c.call i)*(tunings i).BK=c.K at he
  by_contra hh
  have hz : steps (params c.call i)=0 := by omega
  rw [hz,Nat.zero_mul] at he
  omega

structure Accesses {α : Type} (p : Parameters) (initial : State α)
    (bounds : RegionName → Nat) : Prop where
  a : ∀ step, step<steps p → ∀ j : AIndex p,
    inputRow p initial j.1<p.rows → aOffset p initial step j<bounds p.input
  q : ∀ step, step<steps p → ∀ j : BIndex p,
    bOffset p initial step j<bounds p.weight
  s : ∀ step, step<steps p → ∀ j : TileIndex [p.blockColumns],
    scaleOffset p initial step j<bounds p.scale
  z : ∀ step, step<steps p → ∀ j : TileIndex [p.blockColumns],
    zeroOffset p initial step j<bounds p.zero
  c : ∀ j : CIndex p, active p initial j → outputOffset p initial j<bounds p.output

theorem accesses {α : Type} (c : RawKernel.Layout) (h : c.Valid) (i : Fin 17)
    (bounds : RegionName → Nat) (w : Windows c bounds) (s : State α) (pid : Nat)
    (hp : pid<KernelSource.count c.call i) :
    Accesses (params c.call i) (atProgram s pid) bounds := by
  let p := params c.call i
  let initial := atProgram s pid
  have hk8 (k : Nat) (hk : k<c.K) : k/8<c.K/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    simpa only [Nat.div_mul_cancel (dvd_trans (by decide : 8∣128) h.k128)] using hk
  have hn8 (n : Nat) (hn : n<c.N) : n/8<c.N/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    simpa only [Nat.div_mul_cancel (dvd_trans (by decide : 8∣256) h.n256)] using hn
  have hg (step : Nat) (hs : step<steps p) : groupIndex p step<c.K/c.group := by
    have eq := Logical.group_eq p (legal c.call (c.valid h) i) step hs ⟨0,(tuning_facts i).2.2.2.2.1⟩
    rw [eq]
    exact Bounds.group_bound c.call (c.valid h) _
      (Logical.k_bound p (legal c.call (c.valid h) i) step hs _)
  constructor
  · intro step hs j live
    have hv := w.a _ _ live (Logical.k_bound p (legal c.call (c.valid h) i) step hs j.2.1)
    have eq : aOffset p initial step j=c.A.addr (inputRow p initial j.1)
        (step*p.blockInner+j.2.1.val) := by
      simp only [aOffset,inputInner,MatmulKernel.View.addr,p,params,RawKernel.Layout.call]
      ring
    rw [eq]
    exact hv
  · intro step hs j
    have hn := column_bound c h i s pid hp j.2.1
    have hk := Logical.k_bound p (legal c.call (c.valid h) i) step hs j.1
    have eq : bOffset p initial step j=c.Q.addr ((step*p.blockInner+j.1.val)/8)
        (inputColumn p initial j.2.1) := by
      rw [Logical.packed_div p (legal c.call (c.valid h) i)]
      simp only [bOffset,inputInner,MatmulKernel.View.addr,p,params,RawKernel.Layout.call]
      ring
    rw [eq]
    exact w.q _ _ (hk8 _ hk) hn
  · intro step hs j
    have eq : scaleOffset p initial step j=c.S.addr (groupIndex p step)
        (inputColumn p initial j.1) := by
      simp only [scaleOffset,MatmulKernel.View.addr,p,params,RawKernel.Layout.call]
      ring
    rw [eq]
    exact w.s _ _ (hg step hs) (column_bound c h i s pid hp j.1)
  · intro step hs j
    have eq : zeroOffset p initial step j=c.Z.addr (groupIndex p step)
        (inputColumn p initial j.1/8) := by
      simp only [zeroOffset,MatmulKernel.View.addr,p,params,RawKernel.Layout.call]
      ring
    rw [eq]
    exact w.z _ _ (hg step hs) (hn8 _ (column_bound c h i s pid hp j.1))
  · intro j live
    have eq : outputOffset p initial j=c.C.addr (inputRow p initial j.1)
        (inputColumn p initial j.2.1) := by
      simp only [outputOffset,inputRow,inputColumn,MatmulKernel.View.addr,p,params,RawKernel.Layout.call]
      ring
    rw [eq]
    exact w.c _ _ live.1 live.2

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOBounds
