import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeRow

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeGrid
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem value_congr (A : Algebra α) (s t : State α) (X W : RegionName)
    (row N B : Nat) (eps : α) (k : Nat) (hk : k<N)
    (hx : s.mem X=t.mem X) (hw : s.mem W=t.mem W) :
    RuntimeRow.value A s X W row N B eps k=RuntimeRow.value A t X W row N B eps k := by
  have hsq : RuntimeRow.square A s X row N B=RuntimeRow.square A t X row N B := by
    funext i; simp [RuntimeRow.square,RuntimeRow.xload,RuntimeRow.read,hx]
  simp [RuntimeRow.value,RuntimeRow.rstd,RuntimeRow.variance,RuntimeRow.xload,RuntimeRow.wload,RuntimeRow.read,hk,hx,hw,hsq]

theorem writes_congr {ι : Type} (xs : List ι) (Y : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v w : ι → Cell α)
    (s t : State α) (hm : s.mem=t.mem) (hv : ∀ i, active i → v i=w i) :
    (xs.foldl (fun st i => if active i then st.write Y (off i) (v i) else st) s).mem =
      (xs.foldl (fun st i => if active i then st.write Y (off i) (w i) else st) t).mem := by
  induction xs generalizing s t with
  | nil => exact hm
  | cons i xs ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases h : active i
    · simp [h,State.write,hm,hv i h]
    · simp [h,hm]

theorem block_congr (A : Algebra α) (s t : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : α) (h : s.mem=t.mem) :
    (RuntimeRow.block A s X Y W row N B eps).mem=(RuntimeRow.block A t X Y W row N B eps).mem := by
  apply writes_congr _ _ _ _ _ _ s t h
  intro i hi
  rw [value_congr A s t X W row N B eps i.1.val hi (congrFun h X) (congrFun h W)]

theorem block_read (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : α) (hNB : N≤B) (k : Nat) (hk : k<N) :
    (RuntimeRow.block A s X Y W row N B eps).mem Y (row*N+k)=
      .mk .real (RuntimeRow.value A s X W row N B eps k) := by
  let i : TileIndex [B] := ⟨⟨k,lt_of_lt_of_le hk hNB⟩,PUnit.unit⟩
  exact State.masked_scatter_readback Y (fun i : TileIndex [B] => row*N+i.1.val)
    (fun i => Cell.mk .real (RuntimeRow.value A s X W row N B eps i.1.val))
    (fun i => i.1.val<N) s i hk (by
      intro a b h
      have hval : a.1=b.1 := Fin.ext (Nat.add_left_cancel h)
      cases a; cases b; simp_all)

theorem block_frame (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : α) (R : RegionName) (o : Nat)
    (h : R≠Y ∨ ∀ k, k<N → o≠row*N+k) :
    (RuntimeRow.block A s X Y W row N B eps).mem R o=s.mem R o := by
  apply State.masked_scatter_frame
  rcases h with h|h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i.1.val hi)

def setup (rows row : Nat) (s : State α) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun a => if a=0 then row else 0
          numPids := fun a => if a=0 then rows else 1}

/-- Source launches each canonical row, preserving the actual preceding memory. -/
def run (A : Algebra α) (X Y W : RegionName) (rows N B : Nat) (eps : α) (initial : State α) :
    Nat → Option (State α)
  | 0 => some initial
  | k+1 => (run A X Y W rows N B eps initial k).bind
      (fun s => FP.Structural.exec A (runtimeKernel X Y W N B) ((setup rows k s).setReg "eps_arg" .real [] (fun _=>eps)))

def graph (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α) (initial : State α) : Nat → State α
  | 0 => initial
  | k+1 => RuntimeRow.block A (graph A X Y W N B eps initial k) X Y W k N B eps

theorem complete (A : Algebra α) (X Y W : RegionName) (rows N B : Nat) (eps : α)
    (initial : State α) (hN : 0<N) (hNB : N≤B) (count : Nat) :
    ∃ t, run A X Y W rows N B eps initial count=some t ∧
      t.mem=(graph A X Y W N B eps initial count).mem := by
  induction count with
  | zero => exact ⟨initial,rfl,rfl⟩
  | succ k ih =>
    obtain ⟨s,hs,hm⟩ := ih
    obtain ⟨t,ht,hv⟩ := RuntimeRow.complete A (setup rows k s) X Y W N B eps hN hNB
    refine ⟨t,by simpa [run,hs] using ht,?_⟩
    have hc := block_congr A (setup rows k s) (graph A X Y W N B eps initial k) X Y W k N B eps hm
    exact hv.trans hc

def Frame (Y : RegionName) (count : Nat) (s t : State α) : Prop :=
  ∀ R o, R≠Y ∨ count≤o → t.mem R o=s.mem R o

theorem graph_result (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α)
    (initial : State α) (hNB : N≤B) (hXY : X≠Y) (hWY : W≠Y) (count : Nat) :
    (∀ row, row<count → ∀ k, k<N →
      (graph A X Y W N B eps initial count).mem Y (row*N+k)=
        .mk .real (RuntimeRow.value A initial X W row N B eps k)) ∧
    Frame Y (count*N) initial (graph A X Y W N B eps initial count) := by
  induction count with
  | zero => exact ⟨by intro row hr; omega,by intro R o h; rfl⟩
  | succ c ih =>
    let s := graph A X Y W N B eps initial c
    have hx : s.mem X=initial.mem X := funext (fun o => ih.2 X o (Or.inl hXY))
    have hw : s.mem W=initial.mem W := funext (fun o => ih.2 W o (Or.inl hWY))
    constructor
    · intro row hr k hk
      by_cases he : row=c
      · subst row
        rw [graph,block_read A s X Y W c N B eps hNB k hk,
          value_congr A s initial X W c N B eps k hk hx hw]
      · have hrow : row<c := by omega
        have hb : row*N+k<c*N := by
          calc row*N+k < row*N+N := by omega
               _=(row+1)*N := by simp [Nat.add_mul]
               _≤c*N := Nat.mul_le_mul_right N (by omega)
        rw [graph,block_frame A s X Y W c N B eps Y (row*N+k)
          (Or.inr (fun j _ => by omega))]
        exact ih.1 row hrow k hk
    · intro R o ho
      rw [graph,block_frame A s X Y W c N B eps R o (by
        rcases ho with h|h
        · exact Or.inl h
        · right
          intro k hk
          have he : (c+1)*N=c*N+N := by simp [Nat.add_mul]
          omega)]
      apply ih.2 R o
      rcases ho with h|h
      · exact Or.inl h
      · exact Or.inr (le_trans (Nat.mul_le_mul_right N (by omega : c≤c+1)) h)

#axiomsClean complete
#axiomsClean graph_result
#specNonCircular graph avoiding [run,FP.Structural.exec,runtimeKernel]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeGrid
