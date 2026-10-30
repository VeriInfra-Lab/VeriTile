import bench.tritonbench_g.bmm_chunk_bwd.Source
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealContract
import VeriTile.Triton.Memory.FlattenFrame

/-! Whole-program host composition. Fresh registers and semantic zero-undef do
not initialize `torch.empty_like`: programs carry the actual evolving memory.
Only source-active output cells are specified. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostGrid
open VeriTile.Triton
noncomputable section
open Classical
set_option maxHeartbeats 3000000

structure Pid where
  tile : Nat
  batch : Nat
  chunkHead : Nat
  deriving DecidableEq

structure Dimensions where
  tiles : Nat
  batches : Nat
  chunkHeads : Nat

/-- Exactly the three axes passed to `_bmm_chunk_bwd_kernel[grid]`. -/
def dimensions (BM BN batches nchunks : Nat) (p : RealProgram.Parameters) : Dimensions :=
  ⟨((p.chunk + BM - 1) / BM) * ((p.K + BN - 1) / BN), batches, nchunks * p.groups⟩

def programs (g : Dimensions) : List Pid :=
  ((List.range g.tiles).product ((List.range g.batches).product (List.range g.chunkHeads))).map
    (fun q => ⟨q.1, q.2.1, q.2.2⟩)

theorem mem_programs (g : Dimensions) (q : Pid) :
    q ∈ programs g ↔ q.tile < g.tiles ∧ q.batch < g.batches ∧ q.chunkHead < g.chunkHeads := by
  cases q
  simp [programs]

theorem programs_nodup (g : Dimensions) : (programs g).Nodup := by
  apply List.Nodup.map _ (List.nodup_range.product (List.nodup_range.product List.nodup_range))
  rintro ⟨a,b,c⟩ ⟨d,e,f⟩ h
  simpa only [Pid.mk.injEq, Prod.mk.injEq] using h

/-- Fresh program-local state, retaining the previous program's memory. -/
def launch (g : Dimensions) (q : Pid) (s : BlockState) : BlockState where
  mem := s.mem
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then q.tile else if ax = 1 then q.batch else if ax = 2 then q.chunkHead else 0
  numPids := fun ax => if ax = 0 then g.tiles else if ax = 1 then g.batches else if ax = 2 then g.chunkHeads else 1
  undef := fun _ _ => 0

def run (kernel : Kernel) (g : Dimensions) : List Pid → BlockState → Option BlockState
  | [], s => some s
  | q :: rest, s => do
      let next ← exec kernel (launch g q s)
      run kernel g rest next

def io (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :=
  RealViews.io swapped BM BN BCS p

def active (BM BN : Nat) (p : RealProgram.Parameters) (q : Pid) (i : TileIndex [BM, BN]) : Prop :=
  RealViews.row BM BN p q.tile i.1 < RealViews.length p q.chunkHead ∧
    RealViews.column BN p q.tile i.2.1 < p.K

def address (BM BN : Nat) (p : RealProgram.Parameters) (q : Pid) (i : TileIndex [BM, BN]) :=
  RealViews.offset BM BN p p.oBatch p.oRow p.oHead p.oColumn q.tile q.batch q.chunkHead i

/-- Independent initial input views, with three independent stride maps. -/
def inputs (BM BN : Nat) (p : RealProgram.Parameters) (s : BlockState) (q : Pid) :
    RealInputs.Inputs BM BN p.chunk where
  dout := fun i k => s.readMem p.Dout ((q.batch * p.dBatch + RealViews.chunk p q.chunkHead * p.dChunk +
    RealViews.head p q.chunkHead * p.dHead) + (RealViews.row BM BN p q.tile i * p.dColumn + k.val * p.dRow))
  a := fun k j => s.readMem p.A ((q.batch * p.aBatch + RealViews.chunk p q.chunkHead * p.chunk * p.aRow +
    RealViews.head p q.chunkHead * p.aHead) + (k.val * p.aRow + RealViews.column BN p q.tile j * p.aColumn))
  residual := fun i j => s.readMem p.Res
    (RealViews.offset BM BN p p.rBatch p.rRow p.rHead p.rColumn q.tile q.batch q.chunkHead (i,j,PUnit.unit))

def expected (BM BN : Nat) (p : RealProgram.Parameters) (initial : BlockState) (q : Pid)
    (i : TileIndex [BM, BN]) : ℝ :=
  (∑ k : Fin p.chunk, if k.val < min p.chunk (p.seqlen - (q.chunkHead / p.groups) * p.chunk)
    then (inputs BM BN p initial q).dout i.1 k * (inputs BM BN p initial q).a k i.2.1 else 0) +
  if p.hasResidual then (inputs BM BN p initial q).residual i.1 i.2.1 else 0

/-- Inputs may alias one another; the output is separate. -/
def InputsSeparate (p : RealProgram.Parameters) : Prop :=
  p.A ≠ p.Db ∧ p.Dout ≠ p.Db ∧ p.Res ≠ p.Db

/-- A geometric layout condition, independent of all numerical values. -/
def SeparateWrites (BM BN : Nat) (p : RealProgram.Parameters) (jobs : List Pid) : Prop :=
  ∀ q ∈ jobs, ∀ q' ∈ jobs, q ≠ q' → ∀ i j, active BM BN p q i → active BM BN p q' j →
    address BM BN p q i ≠ address BM BN p q' j

def Values (BM BN : Nat) (p : RealProgram.Parameters) (jobs : List Pid) (s t : BlockState) : Prop :=
  ∀ q ∈ jobs, ∀ i, active BM BN p q i → t.readMem p.Db (address BM BN p q i) = expected BM BN p s q i

def Frame (BM BN : Nat) (p : RealProgram.Parameters) (jobs : List Pid) (s t : BlockState) : Prop :=
  ∀ r a, (∀ q ∈ jobs, ∀ i, active BM BN p q i → r ≠ p.Db ∨ a ≠ address BM BN p q i) →
    t.mem r a = s.mem r a

private theorem read_of_mem {s t : BlockState} {r : RegionName} {a : Nat}
    (h : t.mem r a = s.mem r a) : t.readMem r a = s.readMem r a := by
  simp only [BlockState.readMem, h]

/-- Input stability is derived from actual source stores. -/
theorem inputs_stable (BM BN : Nat) (p : RealProgram.Parameters) (jobs : List Pid)
    (s t : BlockState) (separate : InputsSeparate p) (frame : Frame BM BN p jobs s t) (q : Pid) :
    inputs BM BN p t q = inputs BM BN p s q := by
  have h (r : RegionName) (hr : r ≠ p.Db) (a : Nat) : t.readMem r a = s.readMem r a :=
    read_of_mem (frame r a (fun _ _ _ _ => Or.inl hr))
  rcases separate with ⟨ha, hd, hr⟩
  unfold inputs
  congr 1 <;> funext i j
  · exact h p.Dout hd _
  · exact h p.A ha _
  · exact h p.Res hr _

private theorem program_run (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (distinct : RealViews.OutputDistinct BM BN p)
    (g : Dimensions) (q : Pid) (s : BlockState) :
    ∃ t, exec (RealProgram.kernel swapped BM BN BCS p).toAlgKernel (launch g q s) = some t ∧
      Values BM BN p [q] s t ∧ Frame BM BN p [q] s t := by
  have pins : RealIO.Pins (io swapped BM BN BCS p) (launch g q s) (inputs BM BN p s q) :=
    ⟨by intros; rfl, by intros; rfl, by intros; rfl⟩
  obtain ⟨t, he, hy⟩ := RealViews.region_executes swapped (launch g q s) BM BN BCS p hBCS distinct _ pins
  refine ⟨t, he, ?_, ?_⟩
  · intro q' hq i hi
    have hq' : q' = q := by simpa using hq
    subst q'
    exact hy.output i hi
  · intro r a h
    exact hy.frame r a (fun i hi => h q (by simp) i hi)

/-- Successful source executions compose, retaining every initial input,
previous output and raw cell outside the union of active output windows. -/
theorem run_correct (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (distinct : RealViews.OutputDistinct BM BN p) (separate : InputsSeparate p)
    (g : Dimensions) (jobs : List Pid) (unique : jobs.Nodup) (writes : SeparateWrites BM BN p jobs)
    (s : BlockState) :
    ∃ t, run (RealProgram.kernel swapped BM BN BCS p).toAlgKernel g jobs s = some t ∧
      Values BM BN p jobs s t ∧ Frame BM BN p jobs s t := by
  induction jobs generalizing s with
  | nil => exact ⟨s, rfl, by simp [Values], by intro r a _; rfl⟩
  | cons q rest ih =>
    have fresh := (List.nodup_cons.mp unique).1
    have tailUnique := (List.nodup_cons.mp unique).2
    have tailWrites : SeparateWrites BM BN p rest := fun a ha b hb =>
      writes a (by simp [ha]) b (by simp [hb])
    obtain ⟨mid, hm, hvm, hfm⟩ := program_run swapped BM BN BCS p hBCS distinct g q s
    obtain ⟨t, ht, hvt, hft⟩ := ih tailUnique tailWrites mid
    refine ⟨t, by simp only [run, hm]; exact ht, ?_, ?_⟩
    · intro q' hq' i hi
      rcases List.mem_cons.mp hq' with heq | htail
      · subst q'
        have keep := hft p.Db (address BM BN p q i) (by
          intro other ho j hj
          right
          exact writes q (by simp) other (by simp [ho]) (by intro eq; exact fresh (eq ▸ ho)) i j hi hj)
        exact (read_of_mem keep).trans (hvm q (by simp) i hi)
      · rw [hvt q' htail i hi]
        unfold expected
        rw [inputs_stable BM BN p [q] s mid separate hfm q']
    · intro r a h
      exact (hft r a (fun q' hq' i hi => h q' (by simp [hq']) i hi)).trans
        (hfm r a (fun q' hq' i hi => h q' (List.mem_cons.mpr (Or.inl (by simpa using hq'))) i hi))

theorem flatten_launch (alloc : FlatAlloc) (g : Dimensions) (q : Pid) (s : BlockState) :
    alloc.flattenState (launch g q s) = launch g q (alloc.flattenState s) := by
  rfl

/-- Finite flat execution composes with the actual previous flat result. -/
theorem run_flatten (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (g : Dimensions) (jobs : List Pid) (alloc : FlatAlloc) (disjoint : alloc.Disjoint)
    (closed : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (bounded : ∀ q ∈ jobs, ∀ s, RealIO.Bounds (io swapped BM BN BCS p) (launch g q s) alloc.extent)
    (s : BlockState) :
    run (alloc.flattenKernel (RealProgram.kernel swapped BM BN BCS p).toAlgKernel) g jobs (alloc.flattenState s) =
      (run (RealProgram.kernel swapped BM BN BCS p).toAlgKernel g jobs s).map alloc.flattenState := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    have safe := RealSafety.safe swapped (launch g q s) BM BN BCS p alloc.extent
      (RealContract.bounds_of_views swapped _ BM BN BCS p alloc.extent (bounded q (by simp) s))
    have bridge := alloc.exec_flatten disjoint closed _ (launch g q s) safe
      (RealSafety.flatten_ok swapped BM BN BCS p) rfl
    rw [flatten_launch] at bridge
    simp only [run, bridge]
    cases he : exec (RealProgram.kernel swapped BM BN BCS p).toAlgKernel (launch g q s) with
    | none => rfl
    | some next =>
      simp only [Option.map_some]
      exact ih (fun q' hq' => bounded q' (by simp [hq'])) next

/-- Complete finite-memory launch theorem, including readback and raw-cell
frame. No execution or final-output hypothesis occurs in the assumptions. -/
theorem flat_correct (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (distinct : RealViews.OutputDistinct BM BN p) (separate : InputsSeparate p)
    (g : Dimensions) (jobs : List Pid) (unique : jobs.Nodup) (writes : SeparateWrites BM BN p jobs)
    (alloc : FlatAlloc) (disjoint : alloc.Disjoint)
    (closed : ∀ r, r ∉ alloc.regions → alloc.extent r = 0) (outAllocated : p.Db ∈ alloc.regions)
    (bounded : ∀ q ∈ jobs, ∀ s, RealIO.Bounds (io swapped BM BN BCS p) (launch g q s) alloc.extent)
    (s : BlockState) :
    ∃ t, run (alloc.flattenKernel (RealProgram.kernel swapped BM BN BCS p).toAlgKernel) g jobs
        (alloc.flattenState s) = some t ∧
      (∀ q ∈ jobs, ∀ i, active BM BN p q i →
        t.readMem alloc.flat (alloc.addr p.Db (address BM BN p q i)) = expected BM BN p s q i) ∧
      (∀ r a, (r ≠ alloc.flat ∨ ∀ q ∈ jobs, ∀ i, active BM BN p q i →
        a ≠ alloc.addr p.Db (address BM BN p q i)) → t.mem r a = (alloc.flattenState s).mem r a) := by
  obtain ⟨t, he, hv, hf⟩ := run_correct swapped BM BN BCS p hBCS distinct separate g jobs unique writes s
  refine ⟨alloc.flattenState t, ?_, ?_, ?_⟩
  · rw [run_flatten swapped BM BN BCS p g jobs alloc disjoint closed bounded s, he, Option.map_some]
  · intro q hq i hi
    have hb : address BM BN p q i < alloc.extent p.Db := (bounded q hq s).output i hi
    rw [alloc.flattenState_readMem disjoint t outAllocated hb]
    exact hv q hq i hi
  · intro r a outside
    apply alloc.flattenState_frame s t
      (fun region offset => ∃ q ∈ jobs, ∃ i, active BM BN p q i ∧ region = p.Db ∧ offset = address BM BN p q i)
    · intro region offset miss
      apply hf region offset
      intro q hq i hi
      by_cases hr : region = p.Db
      · right
        intro ho
        exact miss ⟨q, hq, i, hi, hr, ho⟩
      · exact Or.inl hr
    · rcases outside with hr | ho
      · exact Or.inl hr
      · right
        rintro region offset ⟨q,hq,i,hi,rfl,rfl⟩
        exact ho q hq i hi

/-- Return the output view, retaining its independent layout. -/
def returned (p : RealProgram.Parameters) (s : BlockState) : RegionName × (Nat → ℝ) :=
  (p.Db, s.readMem p.Db)

#specNonCircular expected avoiding [run]
#specNonCircular Values avoiding [run]
#axiomsClean inputs_stable
#axiomsClean run_correct
#axiomsClean run_flatten
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostGrid
