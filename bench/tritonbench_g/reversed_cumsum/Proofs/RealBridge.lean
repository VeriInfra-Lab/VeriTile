import bench.tritonbench_g.reversed_cumsum.Proofs.RealStep

/-! Reuse the existing descending invariant through an observation adapter.
The old unspecified-load oracle is set to zero only in the reference state;
the repaired source evaluates explicit zero loads for every actual oracle. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.RealBridge
open VeriTile.Triton RealProgram
open VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop (carry NoCollision)
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def zeroUndef (u : BlockState) : BlockState := {u with undef := fun _ _ => 0}

/-- The source stores signed indices; the mathematical suffix invariant keeps
its independent natural coordinates, connected here by exact register values. -/
structure Holds (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) : Prop where
  undef : (zeroUndef u).undef = (zeroUndef initial).undef
  count_le : q ≤ Geometry.chunks h c
  feature : u.regs .int [] "i_s" = some (Tile.scalar (f:Int))
  head : u.regs .int [] "i_bh" = some (Tile.scalar (row:Int))
  triangular : u.regs .real [c.BT,c.BT] "m_s" =
    some (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.upperTriTile c.BT)
  accumulated : u.regs .real [BS] "b_z" =
    some (Tile.vec fun j => some (carry (layout h c) (launch row f) a (fun _ => 0) q j))
  input : ∀ off, (zeroUndef u).readMem h.input off = a off
  output : ∀ (t : Nat) (j : Fin BS),
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active (layout h c) (launch row f) t j.val → q*c.BT ≤ t →
    u.mem h.output (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address (layout h c) (launch row f) t j.val) =
      MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.expected (layout h c) (launch row f) a (fun _ => 0) t j)
  frame : BlockState.WriteWithin
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.footprint (layout h c) (launch row f) h.output) initial u

def modelState (row f : Nat) (u : BlockState) : BlockState :=
  ((zeroUndef u).setReg "i_s" .nat [] (Tile.scalar f)).setReg "i_bh" .nat [] (Tile.scalar row)

theorem toModel (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) (inv : Holds initial h c row f a q u) :
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.Holds (zeroUndef initial)
      (layout h c) (launch row f) h.input h.output a q (modelState row f u) := by
  refine ⟨rfl,inv.count_le,?_,?_,?_,?_,?_,inv.output,inv.frame⟩
  · simp [modelState,launch]
  · simp [modelState,launch]
  · simpa [modelState] using inv.triangular
  · simpa [modelState,zeroUndef,layout] using inv.accumulated
  · simpa [modelState] using inv.input

theorem transfer (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u v : BlockState)
    (hu : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.Holds (zeroUndef initial)
      (layout h c) (launch row f) h.input h.output a q u)
    (mem : u.mem=v.mem)
    (feature : v.regs .int [] "i_s"=some (Tile.scalar (f:Int)))
    (head : v.regs .int [] "i_bh"=some (Tile.scalar (row:Int)))
    (triangular : u.regs .real [c.BT,c.BT] "m_s"=v.regs .real [c.BT,c.BT] "m_s")
    (accumulated : u.regs .real [BS] "b_z"=v.regs .real [BS] "b_z") :
    Holds initial h c row f a q v := by
  refine ⟨rfl,hu.count_le,feature,head,
    triangular ▸ hu.triangular,accumulated ▸ hu.accumulated,?_,?_,?_⟩
  · intro off
    simpa only [zeroUndef,BlockState.readMem,←mem] using hu.input off
  · intro t j ha ht
    simpa only [zeroUndef,←mem] using hu.output t j ha ht
  · intro r off hn
    simpa only [zeroUndef,←mem] using hu.frame r off hn

/-- Scatter's full tagged memory depends only on the incoming tagged memory. -/
theorem scatter_mem (p : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram.Layout)
    (l : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram.Launch)
    (o : RegionName) (a padding : Nat → ℝ) (q : Nat) (acc : Fin p.BS → ℝ)
    (u v : BlockState) (mem : u.mem=v.mem) :
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter p l o a padding q acc u).mem=
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter p l o a padding q acc v).mem := by
  unfold VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter
  generalize TileShape.allIndices [p.BT,p.BS]=xs
  induction xs generalizing u v with
  | nil => exact mem
  | cons idx xs ih =>
    simp only [List.foldl_cons]
    split_ifs
    · apply ih
      funext r off
      simp only [BlockState.writeMem_mem,mem]
    · exact ih u v mem

theorem result_mem (h : Call) (c : Config) (row q f : Nat) (a : Nat → ℝ)
    (acc : Fin BS → ℝ) (u v : BlockState) (mem : u.mem=v.mem) :
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result (layout h c) (launch row f)
      h.input h.output a (fun _ => 0) q acc u).mem=
    (RealStep.result h c row q f a acc v).mem := by
  apply scatter_mem
  exact mem

theorem noCollision (h : Call) (c : Config) (row f : Nat) : NoCollision (layout h c) (launch row f) := by
  intro t u i j ht hu eq
  have hx : f*BS+i.val<h.S := ht.2
  have hy : f*BS+j.val<h.S := hu.2
  have he : t*h.S+(f*BS+i.val)=u*h.S+(f*BS+j.val) := by
    simpa [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,layout,launch,Nat.add_assoc] using eq
  obtain ⟨et,es⟩ := Geometry.pair_unique h.S t u _ _ hx hy he
  exact ⟨et,Fin.ext (by omega)⟩

theorem iteration (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (f : Geometry.FeatureBlock h) (a : Nat → ℝ) (distinct : h.input≠h.output)
    (positive : 0<c.BT) (q : Nat) (u : BlockState) (hq : q<Geometry.chunks h c)
    (inv : Holds initial h c row f.val a (q+1) u) :
    ∃ v, stepStmts (RealStep.chunkCode h c) (u.setReg "i_t" .int [] (Tile.scalar (q:Int)))=some v ∧
      Holds initial h c row f.val a q v := by
  let acc := carry (layout h c) (launch row f.val) a (fun _ => 0) (q+1)
  let entry := u.setReg "i_t" .int [] (Tile.scalar (q:Int))
  let v := RealStep.result h c row q f.val a acc entry
  have he := RealStep.executes h c row q f a acc entry (Geometry.chunk_start h c positive q hq)
    (by simpa [entry,Holds,zeroUndef] using inv.feature)
    (by simpa [entry,Holds,zeroUndef] using inv.head)
    (by simp [entry])
    (by intro off; simpa [entry,Holds,zeroUndef] using inv.input off)
    (by simpa [entry,Holds,zeroUndef,layout] using inv.triangular)
    (by simpa [entry,Holds,zeroUndef,acc] using inv.accumulated)
  have modelInv := toModel initial h c row f.val a (q+1) u inv
  let modelEntry := (modelState row f.val u).setReg "i_t" .int [] (Tile.scalar (q:Int))
  obtain ⟨old,oldExec,oldInv⟩ := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.iteration
    (zeroUndef initial) (layout h c) (launch row f.val) h.input h.output a distinct positive
    (noCollision h c row f.val) q (modelState row f.val u) hq modelInv
  have oldHe := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.executes (layout h c) (launch row f.val)
    h.input h.output a (fun _ => 0) q acc modelEntry (fun _ => rfl)
    (by simp [modelEntry,modelState,launch]) (by simp [modelEntry,modelState,launch])
    (by simp [modelEntry]) (by intro off; simpa [modelEntry,modelState] using inv.input off)
    (by simpa [modelEntry,modelState] using inv.triangular) (by simpa [modelEntry,modelState,zeroUndef,acc] using inv.accumulated)
  have same : old=VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result (layout h c) (launch row f.val)
      h.input h.output a (fun _ => 0) q acc modelEntry := Option.some.inj (oldExec.symm.trans oldHe)
  subst old
  refine ⟨v,he,transfer initial h c row f.val a q _ v oldInv (result_mem h c row q f.val a acc modelEntry entry rfl) ?_ ?_ ?_ ?_⟩
  all_goals simp [v,RealStep.result,RealStep.beforeStore,RealStep.indexed,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.beforeStore,modelEntry,modelState,entry,zeroUndef,layout,inv.feature,inv.head]

#print axioms iteration
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealBridge
