import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.NativeControl
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeComplete
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open Native (word)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def accumulated (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (s : State α) (q : FPGrid.Pid c) :=
  let stop := (c.K+c.BK-1)/c.BK
  NativeControl.range (fun block => NativeIO.update M c d s q block.toNat)
    stop (word 0) (word stop) (fun _ => FPValue.zero M,fun _ => FPValue.zero M)

theorem accumulated_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (h : NativeControl.Accepted c) (s : State α) (q : FPGrid.Pid c) :
    accumulated M c d s q=FPValue.accumulated M c d (FPGrid.launch c s q) := by
  unfold accumulated FPValue.accumulated
  simpa only [Nat.sub_zero] using NativeControl.range_eq
    (fun block => NativeIO.update M c d s q block.toNat)
    (FPValue.update M c d (FPGrid.launch c s q)) 0 ((c.K+c.BK-1)/c.BK)
    (Native.loop_bound c h.storage h.nonempty).2 (Nat.zero_le _)
    (by
      intro n _ hn
      have hw : (word n).toNat=n := NativeIO.nat_of_int _ _
        (Native.word_exact _ (lt_trans hn (Native.loop_bound c h.storage h.nonempty).2))
      dsimp only
      rw [hw]
      exact NativeIO.update_eq M c d h.storage h.nonempty (NativeControl.block32 c h) s q n hn)
    (fun _ => FPValue.zero M,fun _ => FPValue.zero M)

def gatedO (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c)
    (a : FPValue.Acc α c) (i : FPValue.OIndex c) :=
  M.binary (some .fp32) .real .mul (a.1 i) (M.unary (some .fp32) .exp (NativeIO.gValue M c s q i.1))

def gatedS (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c)
    (a : FPValue.Acc α c) (i : FPValue.SIndex c) :=
  if (BitVec.ofNat 32 i.2.1.val).toInt ≤ (BitVec.ofNat 32 i.1.val).toInt then
    M.binary (some .fp32) .real .mul (a.2 i)
      (M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub
        (NativeIO.gValue M c s q i.1) (NativeIO.gValue M c s q i.2.1)))
  else FPValue.zero M

def output (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (s : State α)
    (q : FPGrid.Pid c) (scale : ℝ) (i : FPValue.OIndex c) :=
  let a := accumulated M c d s q
  M.cast (some .fp32) .real .fp32
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .add (gatedO M c s q a i)
        (d.sv (fun j => M.cast (some .fp32) .real .real (gatedS M c s q a j)) (NativeIO.vValue M c s q) i))
      (M.literal (some .fp32) .real scale))

theorem output_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (h : NativeControl.Accepted c) (s : State α) (q : FPGrid.Pid c) (scale : ℝ) :
    output M c d s q scale=FPValue.outputValue M c d (FPGrid.launch c s q) scale := by
  obtain ⟨hv,hg⟩ := NativeIO.tail_loads M c h.storage h.nonempty (NativeControl.block32 c h) s q
  funext i
  simp only [output,FPValue.outputValue,accumulated_eq M c d h,gatedO,gatedS,
    FPValue.gatedO,FPValue.gatedS,NativeControl.causal c h,hg,hv]

/-- Native masked stores use the computed BitVec addresses. Inactive pointers
may wrap and are never interpreted as a memory access. -/
def written (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (s : State α)
    (q : FPGrid.Pid c) (scale : ℝ) : State α :=
  (TileShape.allIndices [c.BT,c.BV]).foldl (fun t i => if NativeIO.active c q i then
    t.write "o" (NativeIO.oOffset c q i).toNat (.mk .fp32 (output M c d s q scale i)) else t)
    (FPGrid.launch c s q)

theorem written_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (h : NativeControl.Accepted c) (s : State α) (q : FPGrid.Pid c) (scale : ℝ) :
    written M c d s q scale=FPValue.written M c scale d (FPGrid.launch c s q) := by
  unfold written FPValue.written
  apply congrArg (fun f => (TileShape.allIndices [c.BT,c.BV]).foldl f (FPGrid.launch c s q))
  funext t i
  have ha := NativeIO.active_eq c h.storage h.nonempty (NativeControl.block32 c h) s q i
  by_cases hi : NativeIO.active c q i
  · rw [if_pos hi,if_pos (ha.mp hi),NativeIO.output_offset c h.storage h.nonempty
      (NativeControl.block32 c h) s q i hi,output_eq M c d h]
  · rw [if_neg hi,if_neg (fun hh => hi (ha.mpr hh))]

def grid (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (scale : ℝ) :
    List (FPGrid.Pid c) → State α → State α
  | [],s => s
  | q::rest,s => grid M c d scale rest (written M c d s q scale)

theorem grid_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (h : NativeControl.Accepted c) (scale : ℝ) (jobs : List (FPGrid.Pid c)) (s : State α) :
    grid M c d scale jobs s=FPGrid.reference M c scale d jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih => simp only [grid,written_eq M c d h,ih,FPGrid.reference]

/-- Native i64 loop/control, actual masked loads and stores, and the typed
source execution have the same complete grid memory. The readback target is
the independent FP recurrence over the initial inputs. -/
theorem whole_grid (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (h : NativeControl.Accepted c) (scale : ℝ) (s : State α) :
    ∃ t, FPGrid.run M c scale (FPGrid.programs c) s=some t ∧
      t.mem=(grid M c d scale (FPGrid.programs c) s).mem ∧ FPGrid.Result M c scale d s t := by
  obtain ⟨t,he,hm⟩ := FPGrid.run_memory M c scale d (FPGrid.programs c) s
  obtain ⟨u,hu,hr⟩ := FPGrid.whole_grid M c scale d s
  have hut : u=t := Option.some.inj (hu.symm.trans he)
  subst u
  exact ⟨t,he,by rw [grid_eq M c d h]; exact hm,hr⟩

/-- Staged native dotAcc graph and native integer execution compose before
any local numerical primitive is interpreted. -/
theorem graph_grid (rest : Algebra NativeGraph.Expr) (c : Shape)
    (h : NativeControl.Accepted c) (scale : ℝ) (s : State NativeGraph.Expr) :
    ∃ t, FPGrid.run (NativeGraph.algebra rest) c scale (FPGrid.programs c) s=some t ∧
      t.mem=(grid (NativeGraph.algebra rest) c (NativeGraph.dots rest c) scale (FPGrid.programs c) s).mem ∧
      FPGrid.Result (NativeGraph.algebra rest) c scale (NativeGraph.dots rest c) s t :=
  whole_grid (NativeGraph.algebra rest) c (NativeGraph.dots rest c) h scale s

#specNonCircular output avoiding [FP.Structural.exec,FPGrid.run]
#axiomsClean whole_grid
#axiomsClean graph_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeComplete
