import bench.tritonbench_g.chunk_gla_simple.Sources.NativeComplete

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.NativeABI
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open NativeControl (Accepted extend32)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000

theorem decode32 (n : Nat) (h : n<2^31) : (extend32 n).toNat=n := by
  rw [NativeControl.extend32_exact n h]
  exact Nat.mod_eq_of_lt (lt_trans h (by decide))

/-- Decode the actual signed-i32 get_program_id results after their source
sign extensions. The indices below carry only the ordinary launch bounds. -/
def pid (c : Shape) (h : Accepted c) (q : FPGrid.Pid c) : FPGrid.Pid c :=
  index c
    ⟨(extend32 (launchIndex c q).value).toNat,by rw [decode32 _ (h.grid q).1]; exact (launch_lt c q).1⟩
    ⟨(extend32 (launchIndex c q).chunk).toNat,by rw [decode32 _ (h.grid q).2.1]; exact (launch_lt c q).2.1⟩
    ⟨(extend32 (launchIndex c q).head).toNat,by rw [decode32 _ (h.grid q).2.2]; exact (launch_lt c q).2.2⟩

theorem pid_eq (c : Shape) (h : Accepted c) (q : FPGrid.Pid c) : pid c h q=q := by
  apply launchIndex_injective c
  have h0 := decode32 _ (h.grid q).1
  have h1 := decode32 _ (h.grid q).2.1
  have h2 := decode32 _ (h.grid q).2.2
  dsimp only [launchIndex] at h0 h1 h2
  simp [pid,launchIndex,index,h0,h1,h2]

def lane {n : Nat} (hn : n<2^31) (i : Fin n) : Fin n :=
  ⟨(extend32 i.val).toNat,by rw [decode32 _ (lt_trans i.isLt hn)]; exact i.isLt⟩

theorem lane_eq {n : Nat} (hn : n<2^31) (i : Fin n) : lane hn i=i := by
  apply Fin.ext
  exact decode32 _ (lt_trans i.isLt hn)

def tBound (c : Shape) (h : Accepted c) : c.BT<2^31 := lt_of_le_of_lt h.block (by decide)
def kBound (c : Shape) : c.BK<2^31 := lt_of_le_of_lt (Nat.min_le_right _ _) (by decide)
def vBound (c : Shape) : c.BV<2^31 := lt_of_le_of_lt (Nat.min_le_right _ _) (by decide)

def qi (c : Shape) (h : Accepted c) (i : FPValue.QIndex c) : FPValue.QIndex c :=
  (lane (tBound c h) i.1,lane (kBound c) i.2.1,PUnit.unit)
def ki (c : Shape) (h : Accepted c) (i : FPValue.KIndex c) : FPValue.KIndex c :=
  (lane (kBound c) i.1,lane (tBound c h) i.2.1,PUnit.unit)
def hi (c : Shape) (h : Accepted c) (i : FPValue.HIndex c) : FPValue.HIndex c :=
  (lane (kBound c) i.1,lane (vBound c) i.2.1,PUnit.unit)
def oi (c : Shape) (h : Accepted c) (i : FPValue.OIndex c) : FPValue.OIndex c :=
  (lane (tBound c h) i.1,lane (vBound c) i.2.1,PUnit.unit)

@[simp] theorem qi_eq (c : Shape) (h : Accepted c) (i : FPValue.QIndex c) : qi c h i=i := by simp [qi,lane_eq]
@[simp] theorem ki_eq (c : Shape) (h : Accepted c) (i : FPValue.KIndex c) : ki c h i=i := by simp [ki,lane_eq]
@[simp] theorem hi_eq (c : Shape) (h : Accepted c) (i : FPValue.HIndex c) : hi c h i=i := by simp [hi,lane_eq]
@[simp] theorem oi_eq (c : Shape) (h : Accepted c) (i : FPValue.OIndex c) : oi c h i=i := by simp [oi,lane_eq]

variable {α : Type} [Inhabited α]

def update (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) (block : BitVec 64) (a : FPValue.Acc α c) : FPValue.Acc α c :=
  let qv := fun i => NativeIO.qValue M c s (pid c h q) block.toNat (qi c h i)
  let kv := fun i => NativeIO.kValue M c s (pid c h q) block.toNat (ki c h i)
  let hv := fun i => NativeIO.hValue M c s (pid c h q) block.toNat (hi c h i)
  (fun i => M.binary (some .fp32) .real .add (a.1 i) (d.qh qv hv i),
   fun i => M.binary (some .fp32) .real .add (a.2 i) (d.qk qv kv i))

theorem update_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) (block : BitVec 64) :
    update M c d h s q block=NativeIO.update M c d s q block.toNat := by
  funext a
  simp [update,pid_eq,NativeIO.update]

def accumulated (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) :=
  let stop := (c.K+c.BK-1)/c.BK
  NativeControl.range (update M c d h s q) stop (Native.word 0) (Native.word stop)
    (fun _ => FPValue.zero M,fun _ => FPValue.zero M)

theorem accumulated_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) : accumulated M c d h s q=NativeComplete.accumulated M c d s q := by
  have he : update M c d h s q=(fun block => NativeIO.update M c d s q block.toNat) := by
    funext block
    exact update_eq M c d h s q block
  simp only [accumulated,NativeComplete.accumulated,he]

def output (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) (scale : ℝ) (i : FPValue.OIndex c) :=
  let a := accumulated M c d h s q
  let gv := fun j => NativeIO.gValue M c s (pid c h q) (lane (tBound c h) j)
  let vv := fun j => NativeIO.vValue M c s (pid c h q) (oi c h j)
  let bo := M.binary (some .fp32) .real .mul (a.1 i) (M.unary (some .fp32) .exp (gv i.1))
  let bs := fun j : FPValue.SIndex c =>
    if (BitVec.ofNat 32 j.2.1.val).toInt ≤ (BitVec.ofNat 32 j.1.val).toInt then
      M.binary (some .fp32) .real .mul (a.2 j)
        (M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (gv j.1) (gv j.2.1)))
    else FPValue.zero M
  M.cast (some .fp32) .real .fp32 (M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .add bo
      (d.sv (fun j => M.cast (some .fp32) .real .real (bs j)) vv i)) (M.literal (some .fp32) .real scale))

theorem output_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) (scale : ℝ) (i : FPValue.OIndex c) :
    output M c d h s q scale i=NativeComplete.output M c d s q scale i := by
  simp only [output,NativeComplete.output,NativeComplete.gatedO,NativeComplete.gatedS,
    accumulated_eq,pid_eq,lane_eq,oi_eq]

def written (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) (scale : ℝ) : State α :=
  (TileShape.allIndices [c.BT,c.BV]).foldl (fun t i => if NativeIO.active c (pid c h q) (oi c h i) then
    t.write "o" (NativeIO.oOffset c (pid c h q) (oi c h i)).toNat (.mk .fp32 (output M c d h s q scale i)) else t)
    (FPGrid.launch c s q)

theorem written_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (s : State α) (q : FPGrid.Pid c) (scale : ℝ) :
    written M c d h s q scale=NativeComplete.written M c d s q scale := by
  simp only [written,NativeComplete.written,pid_eq,oi_eq,output_eq]

def grid (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c) (scale : ℝ) :
    List (FPGrid.Pid c) → State α → State α
  | [],s => s
  | q::rest,s => grid M c d h scale rest (written M c d h s q scale)

theorem grid_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (scale : ℝ) (jobs : List (FPGrid.Pid c)) (s : State α) :
    grid M c d h scale jobs s=NativeComplete.grid M c d scale jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih => simp only [grid,NativeComplete.grid,written_eq,ih]

/-- Actual ABI extension and arange lane decoding occur in the native
recurrence before every masked input/store. They compose to the complete
executed source and independent initial-memory result. -/
theorem whole_grid (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (h : Accepted c)
    (scale : ℝ) (s : State α) :
    ∃ t, FPGrid.run M c scale (FPGrid.programs c) s=some t ∧
      t.mem=(grid M c d h scale (FPGrid.programs c) s).mem ∧ FPGrid.Result M c scale d s t := by
  simpa only [grid_eq] using NativeComplete.whole_grid M c d h scale s

#axiomsClean whole_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.NativeABI
