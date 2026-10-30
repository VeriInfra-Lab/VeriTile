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

/-- The actual source keeps signed program IDs. Mathematical observations retain
explicit-zero padding and the original suffix specification. -/
structure Holds (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) : Prop where
  undef : u.undef = initial.undef
  count_le : q ≤ Geometry.chunks h c
  feature : u.regs .int [] "i_s" = some (Tile.scalar (f : Int))
  head : u.regs .int [] "i_bh" = some (Tile.scalar (row : Int))
  triangular : u.regs .real [c.BT,c.BT] "m_s" =
    some (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.upperTriTile c.BT)
  accumulated : u.regs .real [BS] "b_z" =
    some (Tile.vec fun j => some (carry (layout h c) (launch row f) a (fun _ => 0) q j))
  input : ∀ off, u.readMem h.input off = a off
  output : ∀ (t : Nat) (j : Fin BS),
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active (layout h c) (launch row f) t j.val →
    q*c.BT ≤ t → u.mem h.output
      (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address (layout h c) (launch row f) t j.val) =
      MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.expected
        (layout h c) (launch row f) a (fun _ => 0) t j)
  frame : BlockState.WriteWithin
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.footprint (layout h c) (launch row f) h.output) initial u

/-- Proof-only view for the previous mathematical invariant. The source does not
execute these register writes; its signed ID facts are checked in `Holds`. -/
def legacyState (row f : Nat) (u : BlockState) : BlockState :=
  ((zeroUndef u).setReg "i_s" .nat [] (Tile.scalar f)).setReg "i_bh" .nat [] (Tile.scalar row)

theorem legacy_holds (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) (inv : Holds initial h c row f a q u) :
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.Holds (zeroUndef initial)
      (layout h c) (launch row f) h.input h.output a q (legacyState row f u) := by
  refine ⟨rfl,inv.count_le,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [legacyState,launch]
  · simp [legacyState,launch]
  · simpa [legacyState,zeroUndef,layout] using inv.triangular
  · simpa [legacyState,zeroUndef,layout] using inv.accumulated
  · intro off; simpa [legacyState,zeroUndef] using inv.input off
  · intro t j ha ht; exact inv.output t j ha ht
  · exact inv.frame

theorem transfer (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u v : BlockState)
    (hu : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.Holds (zeroUndef initial)
      (layout h c) (launch row f) h.input h.output a q u)
    (mem : u.mem=v.mem) (undef : v.undef=initial.undef)
    (feature : v.regs .int [] "i_s"=some (Tile.scalar (f : Int)))
    (head : v.regs .int [] "i_bh"=some (Tile.scalar (row : Int)))
    (triangular : u.regs .real [c.BT,c.BT] "m_s"=v.regs .real [c.BT,c.BT] "m_s")
    (accumulated : u.regs .real [BS] "b_z"=v.regs .real [BS] "b_z") :
    Holds initial h c row f a q v := by
  refine ⟨undef,hu.count_le,feature,head,triangular ▸ hu.triangular,
    accumulated ▸ hu.accumulated,?_,?_,?_⟩
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

theorem result_mem_of_memory (h : Call) (c : Config) (row q f : Nat) (a : Nat → ℝ)
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

private theorem fold_undef {ι : Type} (items : List ι) (o : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (u : BlockState) :
    (items.foldl (fun t i => if active i then t.writeMem o (offset i) (value i) else t) u).undef=u.undef := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;>
      simp only [List.foldl_cons,hi,ite_true,ite_false,ih,BlockState.writeMem_undef]
    all_goals rfl

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
    (by simpa [entry] using inv.feature)
    (by simpa [entry] using inv.head)
    (by simp [entry])
    (by intro off; simpa [entry] using inv.input off)
    (by simpa [entry,layout] using inv.triangular)
    (by simpa [entry,acc] using inv.accumulated)
  have oldInvariant := legacy_holds initial h c row f.val a (q+1) u inv
  obtain ⟨old,oldExec,oldInv⟩ := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.iteration
    (zeroUndef initial) (layout h c) (launch row f.val) h.input h.output a distinct positive
    (noCollision h c row f.val) q (legacyState row f.val u) hq oldInvariant
  let oldEntry := (legacyState row f.val u).setReg "i_t" .int [] (Tile.scalar (q:Int))
  have oldHe := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.executes (layout h c) (launch row f.val)
    h.input h.output a (fun _ => 0) q acc oldEntry (fun _ => rfl)
    (by simp [oldEntry,legacyState,launch]) (by simp [oldEntry,legacyState,launch])
    (by simp [oldEntry]) (by intro off; simpa [oldEntry,legacyState,zeroUndef] using inv.input off)
    (by simpa [oldEntry,legacyState,zeroUndef,layout] using inv.triangular)
    (by simpa [oldEntry,legacyState,zeroUndef,acc,layout] using inv.accumulated)
  have same : old=VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result (layout h c) (launch row f.val)
      h.input h.output a (fun _ => 0) q acc oldEntry := Option.some.inj (oldExec.symm.trans oldHe)
  subst old
  refine ⟨v,he,transfer initial h c row f.val a q _ v oldInv
    (result_mem_of_memory h c row q f.val a acc oldEntry entry rfl) ?_ ?_ ?_ ?_ ?_⟩
  · funext r off
    simpa only [v,RealStep.result,BlockState.setReg_undef,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter,fold_undef,
      RealStep.beforeStore,RealStep.indexed,entry] using congrFun (congrFun inv.undef r) off
  all_goals simp [v,RealStep.result,RealStep.beforeStore,RealStep.indexed,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.result,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.scatter,
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.beforeStore,
    zeroUndef,legacyState,oldEntry,entry,layout,inv.undef,inv.feature,inv.head]

#print axioms iteration
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealBridge
