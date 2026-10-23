import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Contract
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
set_option maxHeartbeats 4000000

abbrev tensorIndex (p : Metadata) (b s g n : Nat) := ((b*seqlen p+s)*groups p+g)*width p+n
abbrev doutIndex (p : Metadata) (b c g k r : Nat) := (((b*chunks p+c)*groups p+g)*chunk p+k)*chunk p+r

theorem mul_add_bound {a b n x : Nat} (ha : a<b) (hx : x<n) : a*n+x<b*n := by
  have := Nat.mul_le_mul_right n (show a+1≤b from ha)
  nlinarith

theorem mul_add_inj {n a b x y : Nat} (hx : x<n) (hy : y<n) (eq : a*n+x=b*n+y) : a=b ∧ x=y := by
  have mod := congrArg (·%n) eq
  simp [Nat.add_mod,Nat.mod_eq_of_lt hx,Nat.mod_eq_of_lt hy] at mod
  exact ⟨by have hn : 0<n := Nat.zero_lt_of_lt hx; nlinarith,mod⟩

theorem dimensions_positive (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) :
    0<batch p ∧ 0<seqlen p ∧ 0<groups p ∧ 0<width p ∧ 0<chunks p := by
  rw [input_size p h] at nonempty
  have b : 0<batch p := Nat.pos_of_ne_zero (by intro z; simp [z] at nonempty)
  have s : 0<seqlen p := Nat.pos_of_ne_zero (by intro z; simp [z] at nonempty)
  have g : 0<groups p := Nat.pos_of_ne_zero (by intro z; simp [z] at nonempty)
  have k : 0<width p := Nat.pos_of_ne_zero (by intro z; simp [z] at nonempty)
  refine ⟨b,s,g,k,?_⟩
  rw [h.shapes.2]
  exact Nat.div_pos (by have := h.chunk; omega) h.chunk

theorem dout_positive (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) :
    0<(p.tensor .dout).numel := by
  have pos := dimensions_positive p h nonempty
  rw [dout_size p h]
  exact Nat.mul_pos (Nat.mul_pos (Nat.mul_pos (Nat.mul_pos pos.1 pos.2.2.2.2) pos.2.2.1) h.chunk) h.chunk

theorem tensor_strides (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) :
    p.tensorStrides .a=canonical p ∧
    (p.hasResidual=Bool.true → p.tensorStrides .residual=canonical p) ∧
    (p.hasOutput=Bool.true → p.tensorStrides .output=canonical p) := by
  refine ⟨(tensorStrideEq_iff _ _).mp ((h.strides.1 .a (by simp [Used]) (by decide)).2 nonempty),?_,?_⟩
  · intro present
    apply (tensorStrideEq_iff _ _).mp
    apply (h.strides.1 .residual (by simp [Used,present]) (by decide)).2
    simpa [VeriTile.Torch.TensorMetadata.numel,h.residual present] using nonempty
  · intro present
    apply (tensorStrideEq_iff _ _).mp
    apply (h.strides.1 .output (by simp [Used,present]) (by decide)).2
    simpa [VeriTile.Torch.TensorMetadata.numel,h.output present] using nonempty

theorem dout_strides (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) :
    p.doutStrides=canonicalDout p :=
  (doutStrideEq_iff _ _).mp (h.strides.2.2 (dout_positive p h nonempty))

def dense (p : Metadata) : HostCall :=
  { call p with
    aStrides := canonical p
    doutStrides := canonicalDout p
    outStrides := canonical p
    residualStrides := canonical p }

theorem group_dense (p : Metadata) : (dense p).groupCount=groups p := group_count p

def row (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (i : Fin c.BM) :=
  q.tile/((width p+c.BN-1)/c.BN)*c.BM+i.val
def column (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (j : Fin c.BN) :=
  q.tile%((width p+c.BN-1)/c.BN)*c.BN+j.val
def slice (p : Metadata) (q : HostGrid.Pid) := q.chunkHead/groups p
def head (p : Metadata) (q : HostGrid.Pid) := q.chunkHead-slice p q*groups p
def limit (p : Metadata) (q : HostGrid.Pid) := min (chunk p) (seqlen p-slice p q*chunk p)
def active (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (i : TileIndex [c.BM,c.BN]) : Prop :=
  row p c q i.1<limit p q ∧ column p c q i.2.1<width p

@[simp] theorem source_group (p : Metadata) : (HostContract.parameters (dense p)).groups=groups p := group_count p
@[simp] theorem source_width (p : Metadata) : (HostContract.parameters (dense p)).K=width p := rfl
@[simp] theorem source_row (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (i : Fin c.BM) :
    RealViews.row c.BM c.BN (HostContract.parameters (dense p)) q.tile i=row p c q i := rfl
@[simp] theorem source_column (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (j : Fin c.BN) :
    RealViews.column c.BN (HostContract.parameters (dense p)) q.tile j=column p c q j := rfl
@[simp] theorem source_slice (p : Metadata) (q : HostGrid.Pid) :
    RealViews.chunk (HostContract.parameters (dense p)) q.chunkHead=slice p q := by simp [RealViews.chunk,slice]
@[simp] theorem source_head (p : Metadata) (q : HostGrid.Pid) :
    RealViews.head (HostContract.parameters (dense p)) q.chunkHead=head p q := by simp [RealViews.head,head]
@[simp] theorem source_limit (p : Metadata) (q : HostGrid.Pid) :
    RealViews.length (HostContract.parameters (dense p)) q.chunkHead=limit p q := by
  simp only [RealViews.length,RealMath.chunkLimit,source_slice]
  rfl

@[simp] theorem source_active (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (i : TileIndex [c.BM,c.BN]) :
    HostGrid.active c.BM c.BN (HostContract.parameters (dense p)) q i ↔ active p c q i := by
  simp [HostGrid.active,active]

theorem head_bound (p : Metadata) (q : HostGrid.Pid) (positive : 0<groups p) : head p q<groups p := by
  have m := Nat.mod_lt q.chunkHead positive
  have d := Nat.mod_add_div q.chunkHead (groups p)
  unfold head slice
  rw [Nat.mul_comm (groups p)] at d
  omega

theorem grid_coordinates (p : Metadata) (c : Tuning) (q : HostGrid.Pid)
    (hq : q∈HostGrid.programs (HostContract.grid (dense p) c)) :
    q.batch<batch p ∧ slice p q<chunks p ∧ head p q<groups p := by
  have bounds := (HostGrid.mem_programs _ _).mp hq
  have ch : q.chunkHead<chunks p*groups p := by
    simpa only [HostContract.grid,HostGrid.dimensions,source_group] using bounds.2.2
  have gp : 0<groups p := by nlinarith
  exact ⟨bounds.2.1,(Nat.div_lt_iff_lt_mul gp).mpr ch,head_bound p q gp⟩

theorem valid_sequence (p : Metadata) (q : HostGrid.Pid) (r : Nat) (hr : r<limit p q) :
    slice p q*chunk p+r<seqlen p := by
  have bound := Nat.min_le_right (chunk p) (seqlen p-slice p q*chunk p)
  unfold limit at hr
  omega

theorem head_stride (p : Metadata) : (dense p).headStride (canonical p).head=groups p*0+width p*Bool.toNat (grouped p) := by
  cases g : grouped p <;> simp [TypedSource.HostCall.headStride,dense,call,canonical,g]

theorem tensor_offset (p : Metadata) (b s g n : Nat) (hg : g<groups p) :
    b*(canonical p).batch+s*(canonical p).sequence+g*((dense p).headStride (canonical p).head)+n*(canonical p).feature=
      tensorIndex p b s g n := by
  cases grouped : grouped p
  · have zero : g=0 := by simpa [groups,grouped] using hg
    subst g
    simp [canonical,dense,call,TypedSource.HostCall.headStride,grouped,tensorIndex,groups]
    ring
  · simp only [canonical,dense,call,TypedSource.HostCall.headStride,grouped,↓reduceIte,Nat.mul_one]
    ring

theorem output_offset (p : Metadata) (c : Tuning) (q : HostGrid.Pid) (i : TileIndex [c.BM,c.BN])
    (hg : head p q<groups p) :
    HostGrid.address c.BM c.BN (HostContract.parameters (dense p)) q i=
      tensorIndex p q.batch (slice p q*chunk p+row p c q i.1) (head p q) (column p c q i.2.1) := by
  unfold HostGrid.address RealViews.offset
  simp only [source_slice,source_head,source_row,source_column]
  change (q.batch*(canonical p).batch+slice p q*chunk p*(canonical p).sequence+
    head p q*((dense p).headStride (canonical p).head))+
      (row p c q i.1*(canonical p).sequence+column p c q i.2.1*(canonical p).feature)=_
  rw [←tensor_offset p _ _ _ _ hg]
  ring

theorem parameters_eq (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) :
    HostContract.parameters (call p)=HostContract.parameters (dense p) := by
  obtain ⟨ha,hr,ho⟩ := tensor_strides p h nonempty
  have hd := dout_strides p h nonempty
  have hout : (if p.hasOutput then p.tensorStrides .output else canonical p)=canonical p := by
    cases hh : p.hasOutput <;> simp [hh,ho]
  cases res : p.hasResidual
  · simp [HostContract.parameters,dense,call,TypedSource.HostCall.resStride,TypedSource.HostCall.groupCount,TypedSource.HostCall.headStride,res,ha,hd,hout]
  · have hres := hr res
    simp [HostContract.parameters,dense,call,TypedSource.HostCall.resStride,res,ha,hd,hout,hres]

theorem source_eq (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) (c : Tuning) :
    (call p).source c Bool.false=(dense p).source c Bool.false := by
  rw [HostContract.source_eq,HostContract.source_eq,parameters_eq p h nonempty]
  rfl

theorem grid_eq (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) (c : Tuning) :
    HostContract.grid (call p) c=HostContract.grid (dense p) c := by
  simp only [HostContract.grid,parameters_eq p h nonempty]
  rfl

theorem tensor_bound (p : Metadata) (h : InputContract p) (b s g n : Nat)
    (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    tensorIndex p b s g n<(p.tensor .a).numel := by
  rw [input_size p h]
  exact mul_add_bound (mul_add_bound (mul_add_bound hb hs) hg) hn

theorem dout_bound (p : Metadata) (h : InputContract p) (b c g k r : Nat)
    (hb : b<batch p) (hc : c<chunks p) (hg : g<groups p) (hk : k<chunk p) (hr : r<chunk p) :
    doutIndex p b c g k r<(p.tensor .dout).numel := by
  rw [dout_size p h]
  exact mul_add_bound (mul_add_bound (mul_add_bound (mul_add_bound hb hc) hg) hk) hr

theorem tensor_injective (p : Metadata) (b s g n b' s' g' n' : Nat)
    (hs : s<seqlen p) (hs' : s'<seqlen p) (hg : g<groups p) (hg' : g'<groups p)
    (hn : n<width p) (hn' : n'<width p)
    (eq : tensorIndex p b s g n=tensorIndex p b' s' g' n') : b=b' ∧ s=s' ∧ g=g' ∧ n=n' := by
  obtain ⟨e1,en⟩ := mul_add_inj hn hn' eq
  obtain ⟨e2,eg⟩ := mul_add_inj hg hg' e1
  obtain ⟨eb,es⟩ := mul_add_inj hs hs' e2
  exact ⟨eb,es,eg,en⟩

theorem a_offset (p : Metadata) (c : Tuning) (q : HostGrid.Pid)
    (i : TileIndex [chunk p,c.BN]) (hg : head p q<groups p) :
    (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).aWindow q.tile q.batch q.chunkHead i=
      tensorIndex p q.batch (slice p q*chunk p+i.1.val) (head p q) (column p c q i.2.1) := by
  unfold HostGrid.io RealViews.io
  simp only [source_slice,source_head,source_column]
  change (q.batch*(canonical p).batch+slice p q*chunk p*(canonical p).sequence+
    head p q*((dense p).headStride (canonical p).head))+
      (i.1.val*(canonical p).sequence+column p c q i.2.1*(canonical p).feature)=_
  rw [←tensor_offset p _ _ _ _ hg]
  ring

theorem dout_offset (p : Metadata) (c : Tuning) (q : HostGrid.Pid)
    (i : TileIndex [c.BM,chunk p]) (hg : head p q<groups p) :
    (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).doutWindow q.tile q.batch q.chunkHead i=
      doutIndex p q.batch (slice p q) (head p q) i.2.1.val (row p c q i.1) := by
  unfold HostGrid.io RealViews.io
  simp only [source_slice,source_head,source_row]
  cases grouped : grouped p
  · have zero : head p q=0 := by simpa [groups,grouped] using hg
    simp [HostContract.parameters,canonicalDout,dense,call,TypedSource.HostCall.headStride,grouped,zero,doutIndex]
    ring
  · simp [HostContract.parameters,canonicalDout,dense,call,TypedSource.HostCall.headStride,grouped,doutIndex]
    ring

theorem residual_offset (p : Metadata) (c : Tuning) (q : HostGrid.Pid)
    (i : TileIndex [c.BM,c.BN]) (hg : head p q<groups p) (present : p.hasResidual=Bool.true) :
    (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).residualWindow q.tile q.batch q.chunkHead i=
      tensorIndex p q.batch (slice p q*chunk p+row p c q i.1) (head p q) (column p c q i.2.1) := by
  have same : (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).residualWindow q.tile q.batch q.chunkHead i=
      HostGrid.address c.BM c.BN (HostContract.parameters (dense p)) q i := by
    simp [HostGrid.io,RealViews.io,HostGrid.address,HostContract.parameters,dense,call,TypedSource.HostCall.resStride,present]
  rw [same,output_offset p c q i hg]

theorem output_distinct (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel) (c : Tuning) :
    RealViews.OutputDistinct c.BM c.BN (HostContract.parameters (dense p)) := by
  intro pid0 pid1 pid2 i j hi hj eq
  have pos := dimensions_positive p h nonempty
  have stride : width p≤groups p*width p := by nlinarith [pos.2.2.1]
  have ci : RealViews.column c.BN (HostContract.parameters (dense p)) pid0 i.2.1<groups p*width p :=
    lt_of_lt_of_le hi.2 stride
  have cj : RealViews.column c.BN (HostContract.parameters (dense p)) pid0 j.2.1<groups p*width p :=
    lt_of_lt_of_le hj.2 stride
  unfold RealViews.offset at eq
  have roweq : RealViews.row c.BM c.BN (HostContract.parameters (dense p)) pid0 i.1=
      RealViews.row c.BM c.BN (HostContract.parameters (dense p)) pid0 j.1 := by
    have e : RealViews.row c.BM c.BN (HostContract.parameters (dense p)) pid0 i.1*(groups p*width p)+
        RealViews.column c.BN (HostContract.parameters (dense p)) pid0 i.2.1=
        RealViews.row c.BM c.BN (HostContract.parameters (dense p)) pid0 j.1*(groups p*width p)+
        RealViews.column c.BN (HostContract.parameters (dense p)) pid0 j.2.1 := by
      simpa only [HostContract.parameters,dense,canonical,Nat.mul_one,Nat.add_right_inj] using eq
    exact (mul_add_inj ci cj e).1
  have ie : i.1=j.1 := Fin.ext (by simpa [RealViews.row] using roweq)
  have je : i.2.1=j.2.1 := Fin.ext (by
    simpa only [ie,HostContract.parameters,dense,canonical,Nat.mul_one,Nat.add_right_inj,RealViews.column] using eq)
  exact Prod.ext ie (Prod.ext je (Subsingleton.elim _ _))

theorem program_writes (p : Metadata) (h : InputContract p) (c : Tuning) :
    HostGrid.SeparateWrites c.BM c.BN (HostContract.parameters (dense p))
      (HostGrid.programs (HostContract.grid (dense p) c)) := by
  intro q hq q' hq' different i j hi hj eq
  have gq := grid_coordinates p c q hq
  have gq' := grid_coordinates p c q' hq'
  have ai := (source_active p c q i).mp hi
  have aj := (source_active p c q' j).mp hj
  rw [output_offset p c q i gq.2.2,output_offset p c q' j gq'.2.2] at eq
  obtain ⟨eb,es,eg,en⟩ := tensor_injective p _ _ _ _ _ _ _ _ (valid_sequence p q _ ai.1)
    (valid_sequence p q' _ aj.1) gq.2.2 gq'.2.2 ai.2 aj.2 eq
  have ri : row p c q i.1<chunk p := lt_of_lt_of_le ai.1 (Nat.min_le_left _ _)
  have rj : row p c q' j.1<chunk p := lt_of_lt_of_le aj.1 (Nat.min_le_left _ _)
  obtain ⟨ec,er⟩ := mul_add_inj ri rj es
  have ech : q.chunkHead=q'.chunkHead := by
    have d := Nat.div_add_mod' q.chunkHead (groups p)
    have d' := Nat.div_add_mod' q'.chunkHead (groups p)
    dsimp only [head,slice] at eg ec
    rw [ec] at eg d
    omega
  have erow := (mul_add_inj i.1.isLt j.1.isLt er).1
  have ecol := (mul_add_inj i.2.1.isLt j.2.1.isLt en).1
  have etile : q.tile=q'.tile := by
    have d := Nat.div_add_mod' q.tile ((width p+c.BN-1)/c.BN)
    have d' := Nat.div_add_mod' q'.tile ((width p+c.BN-1)/c.BN)
    change q.tile/((width p+c.BN-1)/c.BN)=q'.tile/((width p+c.BN-1)/c.BN) at erow
    change q.tile%((width p+c.BN-1)/c.BN)=q'.tile%((width p+c.BN-1)/c.BN) at ecol
    rw [erow,ecol] at d
    exact d.symm.trans d'
  apply different
  cases q; cases q'
  simp_all

#axiomsClean parameters_eq
#axiomsClean source_eq
#axiomsClean output_distinct
#axiomsClean program_writes
#axiomsClean grid_coordinates
#axiomsClean output_offset
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Direct
