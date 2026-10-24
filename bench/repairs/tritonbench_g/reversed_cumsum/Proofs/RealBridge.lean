import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealStep

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

def Holds (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) : Prop :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.Holds (zeroUndef initial)
    (layout h c) (launch row f) h.input h.output a q (zeroUndef u)

theorem transfer (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u v : BlockState)
    (hu : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.Holds (zeroUndef initial)
      (layout h c) (launch row f) h.input h.output a q u)
    (mem : u.mem=v.mem)
    (feature : u.regs .nat [] "i_s"=v.regs .nat [] "i_s")
    (head : u.regs .nat [] "i_bh"=v.regs .nat [] "i_bh")
    (triangular : u.regs .real [c.BT,c.BT] "m_s"=v.regs .real [c.BT,c.BT] "m_s")
    (accumulated : u.regs .real [BS] "b_z"=v.regs .real [BS] "b_z") :
    Holds initial h c row f a q v := by
  refine ⟨rfl,hu.count_le,feature ▸ hu.feature,head ▸ hu.head,
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
    (acc : Fin BS → ℝ) (u : BlockState) :
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result (layout h c) (launch row f)
      h.input h.output a (fun _ => 0) q acc (zeroUndef u)).mem=
    (RealStep.result h c row q f a acc u).mem := by
  apply scatter_mem
  rfl

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
  obtain ⟨old,oldExec,oldInv⟩ := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.iteration
    (zeroUndef initial) (layout h c) (launch row f.val) h.input h.output a distinct positive
    (noCollision h c row f.val) q (zeroUndef u) hq inv
  have oldHe := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.executes (layout h c) (launch row f.val)
    h.input h.output a (fun _ => 0) q acc (zeroUndef entry) (fun _ => rfl)
    (by simpa [entry,zeroUndef] using inv.feature) (by simpa [entry,zeroUndef] using inv.head)
    (by simp [entry,zeroUndef]) (by intro off; simpa [entry,zeroUndef] using inv.input off)
    (by simpa [entry,zeroUndef] using inv.triangular) (by simpa [entry,zeroUndef,acc] using inv.accumulated)
  have same : old=VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result (layout h c) (launch row f.val)
      h.input h.output a (fun _ => 0) q acc (zeroUndef entry) := Option.some.inj (oldExec.symm.trans oldHe)
  subst old
  refine ⟨v,he,transfer initial h c row f.val a q _ v oldInv (result_mem h c row q f.val a acc entry) ?_ ?_ ?_ ?_⟩
  all_goals simp [v,RealStep.result,RealStep.beforeStore,RealStep.indexed,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.beforeStore,zeroUndef,layout]

#print axioms iteration
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealBridge
