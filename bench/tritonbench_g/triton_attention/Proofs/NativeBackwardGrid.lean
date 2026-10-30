import bench.tritonbench_g.triton_attention.Proofs.NativeBackwardComplete
import bench.tritonbench_g.triton_attention.Proofs.NativeGraphMemory
import bench.tritonbench_g.triton_attention.Proofs.GridUnique
import bench.tritonbench_g.triton_attention.Proofs.BackwardGridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000

def TensorData (c : Shape) (mem : Grid.Memory Graph) (r : RegionName) (x : Target.Tensor) : Prop :=
  ∀ i : Coord c,mem r (offset c i)=.mk .fp32 (x i.1.val i.2.1.val i.2.2.val)
def RowData (c : Shape) (mem : Grid.Memory Graph) (r : RegionName) (x : Target.Rows) : Prop :=
  ∀ h : Fin c.groups,∀ i : Fin c.rows,mem r (rowOffset c h i)=.mk .fp32 (x h.val i.val)
structure BackwardData (c : Shape) (mem : Grid.Memory Graph) (x : Target.BackwardInputs) : Prop where
  q : TensorData c mem "Q" x.q
  k : TensorData c mem "K" x.k
  v : TensorData c mem "V" x.v
  gradient : TensorData c mem "NewDO" x.gradient
  maximum : RowData c mem "M" x.maximum
  delta : RowData c mem "Delta" x.delta

def HeadZero (c : Shape) (head : Fin c.groups) (mem : Grid.Memory Graph) : Prop :=
  ∀ i : Coord c,i.1=head → mem "DQ" (offset c i)=.mk .fp32 (.word 0)

theorem blockRow_bound (c : Shape) (b : Nat) (hb : b<c.blocks) (i : Fin 128) : b*128+i.val<c.rows := by
  have hi:=i.isLt
  change _<c.blocks*128
  omega

theorem entry_backwardReady (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (x : Target.BackwardInputs) (h : BackwardData c mem x) (hp : Grid.Legal (Grid.backwardGrid c) pid) :
    BackwardReady c scale (Grid.entry (Grid.backwardGrid c) pid (embed mem)) (embed mem) x pid.1 := by
  have read (r : RegionName) (f : Target.Tensor) (hr : TensorData c mem r f) (b : Nat) (hb : b<c.blocks)
      (i : BLane (backwardParameters c scale)) :
      backwardRead (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid (embed mem)) (embed mem) r b i=
        fun _=>Stage.ready (f pid.1 (b*128+i.1.val) i.2.1.val) := by
    unfold backwardRead
    rw [pointer_inBounds c scale _ r b (by intro j;rw [Grid.backwardHead_eq];exact Grid.row_bound c pid.1 b j.val hp.1 hb j.isLt),if_pos rfl,
      Grid.backwardOffset_eq]
    have hm:=hr (⟨pid.1,hp.1⟩,⟨b*128+i.1.val,blockRow_bound c b hb i.1⟩,i.2.1)
    simp only [offset,Nat.add_assoc] at hm
    simp only [embed,Nat.add_assoc,hm]
    rfl
  have rows (r : RegionName) (f : Target.Rows) (hr : RowData c mem r f) (b : Nat) (hb : b<c.blocks) (i : Fin 128) :
      ((embed mem) r (backwardRowOffset (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid (embed mem)) b (i,()))).read .fp32=
        fun _=>Stage.ready (f pid.1 (b*128+i.val)) := by
    have hm:=hr ⟨pid.1,hp.1⟩ ⟨b*128+i.val,blockRow_bound c b hb i⟩
    simp only [rowOffset] at hm
    have hoff : backwardRowOffset (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid (embed mem)) b (i,())=
        pid.1*c.rows+(b*128+i.val) := by simp [backwardRowOffset,backwardParameters,Grid.entry,Grid.ids,Shape.rows,Nat.mul_comm,Nat.add_assoc]
    rw [hoff]
    simp only [embed,hm]
    rfl
  exact ⟨read "Q" x.q h.q,read "K" x.k h.k,read "V" x.v h.v,read "NewDO" x.gradient h.gradient,
    rows "M" x.maximum h.maximum,rows "Delta" x.delta h.delta⟩

theorem backwardStep_readonly (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (r : RegionName) (o : Nat) (hr : r≠"DQ" ∧ r≠"DK" ∧ r≠"DV") :
    store one .backward c scale owners pid mem r o=mem r o := by
  change mapCell _ (Grid.backwardStep (modelScalar one .backward) c scale pid (embed mem) r o)=_
  rw [Grid.backwardStep_readonly _ c scale pid (embed mem) r o hr]
  exact congrFun (congrFun (finish_embed one .backward c owners mem) r) o

theorem backwardStep_data (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (x : Target.BackwardInputs) (h : BackwardData c mem x) :
    BackwardData c (store one .backward c scale owners pid mem) x := by
  constructor
  · intro i;rw [backwardStep_readonly one c scale owners pid mem "Q" _ (by decide)];exact h.q i
  · intro i;rw [backwardStep_readonly one c scale owners pid mem "K" _ (by decide)];exact h.k i
  · intro i;rw [backwardStep_readonly one c scale owners pid mem "V" _ (by decide)];exact h.v i
  · intro i;rw [backwardStep_readonly one c scale owners pid mem "NewDO" _ (by decide)];exact h.gradient i
  · intro hd i;rw [backwardStep_readonly one c scale owners pid mem "M" _ (by decide)];exact h.maximum hd i
  · intro hd i;rw [backwardStep_readonly one c scale owners pid mem "Delta" _ (by decide)];exact h.delta hd i

theorem backwardStep_other (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (Grid.backwardGrid c) pid) (i : Coord c) (hn : pid≠Grid.backwardProgram c i) (r : RegionName) :
    store one .backward c scale owners pid mem r (offset c i)=mem r (offset c i) := by
  change mapCell _ (Grid.backwardStep (modelScalar one .backward) c scale pid (embed mem) r (offset c i))=_
  rw [Grid.backwardStep_other _ c scale pid (embed mem) hp i hn r]
  exact congrFun (congrFun (finish_embed one .backward c owners mem) r) (offset c i)

theorem backwardStep_headZero (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (Grid.backwardGrid c) pid) (target : Coord c) (hne : pid≠Grid.backwardProgram c target)
    (h : HeadZero c target.1 mem) : HeadZero c target.1 (store one .backward c scale owners pid mem) := by
  intro i hi
  rw [backwardStep_other one c scale owners pid mem hp i (by simpa [Grid.backwardProgram,hi] using hne) "DQ"]
  exact h i hi

theorem represents_finish (one : Bool) (c : Shape) (owners : TritonAttention.Whole001.Native.Owners)
    (mem : Grid.Memory Expr) (r : RegionName) (o : Nat) (g : Graph) (h : Represents one (mem r o) g) :
    finish one .backward c owners mem r o=.mk .fp32 g := by
  obtain ⟨e,he,hg⟩:=h
  change mapCell _ (mem r o)=_
  rw [he]
  change Cell.mk .fp32 (lowerScalar one .backward (e (storeOwner c owners r o)))=_
  rw [hg]

theorem backwardStep_values (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (mem : Grid.Memory Graph)
    (x : Target.BackwardInputs) (h : BackwardData c mem x) (i : Coord c) (hz : HeadZero c i.1 mem) :
    store one .backward c scale owners (Grid.backwardProgram c i) mem "DQ" (offset c i)=
      .mk .fp32 (Target.query c one scale x i.1.val i.2.1.val i.2.2.val c.blocks) ∧
    store one .backward c scale owners (Grid.backwardProgram c i) mem "DK" (offset c i)=
      .mk .fp32 (Target.key c one scale x i.1.val i.2.1.val i.2.2.val) ∧
    store one .backward c scale owners (Grid.backwardProgram c i) mem "DV" (offset c i)=
      .mk .fp32 (Target.value c one scale x i.1.val i.2.1.val i.2.2.val) := by
  let pid:=Grid.backwardProgram c i
  let s:=Grid.entry (Grid.backwardGrid c) pid (embed mem)
  have hp:=Grid.backwardProgram_legal c i
  have hb : ∀ b<c.blocks,∀ j : Fin 128,backwardHeadRow (backwardParameters c scale) s+b*128+j.val<(backwardParameters c scale).rows := by
    intro b hbn j
    dsimp only [s]
    rw [Grid.backwardHead_eq]
    exact Grid.row_bound c pid.1 b j.val hp.1 hbn j.isLt
  have hq : ∀ b,b<c.blocks → ∀ lane : BLane (backwardParameters c scale),
      Represents one (s.mem "DQ" (backwardOffset (backwardParameters c scale) s b lane)) (.word 0) := by
    intro b hbn lane
    have hm:=hz (i.1,⟨b*128+lane.1.val,blockRow_bound c b hbn lane.1⟩,lane.2.1) rfl
    dsimp only [s]
    rw [Grid.backwardOffset_eq]
    simp only [offset,Nat.add_assoc] at hm
    simp only [pid,Grid.backwardProgram,Grid.entry,embed,Nat.add_assoc,hm]
    exact ⟨fun _=>Stage.ready (.word 0),rfl,fun _=>rfl⟩
  obtain ⟨final,he,ho,_⟩:=backward_execution_native c scale one s x pid.1
    (entry_backwardReady c scale pid mem x h hp) hb hq
  change FP.Structural.exec (backend (algebraScalar one .backward) (modelScalar one .backward)) (backwardKernel c scale)
    (Grid.entry (Grid.backwardGrid c) pid (embed mem))=some final at he
  have hm : final.mem=Grid.backwardStep (modelScalar one .backward) c scale pid (embed mem) := by
    simpa only [Grid.execute,he,Option.map_some,Option.some.injEq] using
      Grid.backward_success (algebraScalar one .backward) (modelScalar one .backward) c scale pid (embed mem)
  have ans:=ho (i.2.1.val/128) (grid_coverage c i.2.1).1 (Grid.backwardLane c scale i)
  rw [hm] at ans
  dsimp only [s,pid] at ans
  rw [Grid.backward_target_offset] at ans
  simp only [Grid.backwardProgram,Grid.backwardLane,Nat.div_add_mod'] at ans
  exact ⟨represents_finish one c owners _ "DQ" _ _ ans.1,
    represents_finish one c owners _ "DK" _ _ ans.2.1,
    represents_finish one c owners _ "DV" _ _ ans.2.2⟩

theorem backward_grid (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.backwardGrid c) work) (mem : Grid.Memory Graph)
    (x : Target.BackwardInputs) (h : BackwardData c mem x)
    (hz : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (.word 0)) :
    ∃ final,Grid.run (execute one .backward c scale owners) work mem=some final ∧ BackwardData c final x ∧
      ∀ i : Coord c,
        final "DQ" (offset c i)=.mk .fp32 (Target.query c one scale x i.1.val i.2.1.val i.2.2.val c.blocks) ∧
        final "DK" (offset c i)=.mk .fp32 (Target.key c one scale x i.1.val i.2.1.val i.2.2.val) ∧
        final "DV" (offset c i)=.mk .fp32 (Target.value c one scale x i.1.val i.2.1.val i.2.2.val) := by
  let final:=work.foldl (fun m pid=>store one .backward c scale owners pid m) mem
  have hlegal:=Grid.FullSchedule.legal (Grid.backwardGrid c) work schedule
  have hnodup:=Grid.schedule_nodup (Grid.backwardGrid c) work schedule
  have hinv : BackwardData c final x :=
    Grid.fold_invariant (store one .backward c scale owners) (fun m=>BackwardData c m x) work mem h
      (by intro pid hp m hm;exact backwardStep_data one c scale owners pid m x hm)
  refine ⟨final,Grid.run_eq_fold _ _ (execute_success one .backward c scale owners) work mem,hinv,?_⟩
  intro i
  have aux (r : RegionName) (g : Graph)
      (hw : ∀ m,BackwardData c m x → HeadZero c i.1 m →
        store one .backward c scale owners (Grid.backwardProgram c i) m r (offset c i)=.mk .fp32 g) :
      final r (offset c i)=.mk .fp32 g := by
    apply Grid.fold_readback_unique (store one .backward c scale owners)
      (fun m=>BackwardData c m x ∧ HeadZero c i.1 m) work mem (Grid.backwardProgram c i) r (offset c i) _ hnodup
      (Grid.FullSchedule.covers _ work schedule _ (Grid.backwardProgram_legal c i)) ⟨h,by intro j hj;exact hz j⟩
    · intro pid hp hn m hm
      exact ⟨backwardStep_data one c scale owners pid m x hm.1,backwardStep_headZero one c scale owners pid m (hlegal pid hp) i hn hm.2⟩
    · intro m hm
      exact hw m hm.1 hm.2
    · intro pid hp hn m
      exact backwardStep_other one c scale owners pid m (hlegal pid hp) i hn r
  exact ⟨aux "DQ" _ (by intro m hm hzero;exact (backwardStep_values one c scale owners m x hm i hzero).1),
    aux "DK" _ (by intro m hm hzero;exact (backwardStep_values one c scale owners m x hm i hzero).2.1),
    aux "DV" _ (by intro m hm hzero;exact (backwardStep_values one c scale owners m x hm i hzero).2.2)⟩

#axiomsClean backward_grid
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
