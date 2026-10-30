import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.Execution

namespace VeriTile.Bench.Repairs.ContextAttnMistral.Mathematical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Checked Execution
noncomputable section
set_option maxHeartbeats 4000000

def logical (input : Recurrence.Inputs ℝ) : Logical.Tensors := ⟨input.query,input.key,input.value⟩

/-- Ordinary read-only reification of current tensor coordinates into source
roles. Fields bind only input cells and metadata; no execution/result field.
Virtual role names permit physical read/read aliasing. No Python copies occur. -/
structure Adapter (c : Host.Call) (tokens : Nat) where
  allocation : FlatAlloc
  state : Recurrence.Inputs ℝ → BlockState
  metadata : ∀ input, RoleBinding c tokens allocation (state input)
  reads : ∀ input, Logical.Resolved c (state input) (logical input)

def index (q : Coord c) : GridIndex c.grid :=
  c.indexOf q.val.1 q.val.2.1 ⟨row q/128,(Nat.div_lt_iff_lt_mul (by decide)).mpr q.val.2.2.1.isLt⟩

theorem index_launch (q : Coord c) : c.launchAt (index q)=⟨row q/128,batch q,head q⟩ :=
  Host.launch_indexOf c _ _ _

def stage (adapter : Adapter c tokens) (input : Recurrence.Inputs ℝ) (q : Coord c) : BlockState :=
  (adapter.state input).withGridIndex (index q)

/-- Actual full Real AST body execution followed by its physical store value.
getD is totalization only; source_success below proves it always selects some. -/
def source (adapter : Adapter c tokens) (input : Recurrence.Inputs ℝ) (q : Coord c) : ℝ :=
  ((stepStmts (Program.kernel c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)).eraseDType.toAlgKernel.body
    (stage adapter input q)).getD (stage adapter input q)).readMem c.regions.out (offset q)

def target (c : Host.Call) (input : Recurrence.Inputs ℝ) (q : Coord c) : ℝ :=
  Logical.expected c (logical input) (batch q) (head q) (row q) (dim q)

theorem source_success (g : Geometry c tokens) (adapter : Adapter c tokens)
    (input : Recurrence.Inputs ℝ) (q : Coord c) :
    ∃ final, stepStmts (Program.kernel c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)).eraseDType.toAlgKernel.body
      (stage adapter input q)=some final ∧ final.readMem c.regions.out (offset q)=target c input q := by
  let s := adapter.state input
  let A := adapter.allocation
  have v : Host.Valid c A s := host_valid g (adapter.metadata input)
  have bounds := Host.program_bounds c A s v (index q)
  have injective : ∀ i j, EpilogueResult.activeLane c.layout c.regions (stage adapter input q) i →
      EpilogueResult.activeLane c.layout c.regions (stage adapter input q) j →
      EpilogueResult.offset c.layout c.regions (stage adapter input q) i=
        EpilogueResult.offset c.layout c.regions (stage adapter input q) j → i=j := by
    intro i j hi hj eq
    change RealIO.write c.layout (Program.launch (s.withGridIndex (index q)))
      (Program.inputs (s.withGridIndex (index q)) c.regions) i=
      RealIO.write c.layout (Program.launch (s.withGridIndex (index q)))
      (Program.inputs (s.withGridIndex (index q)) c.regions) j at eq
    rw [Host.offset_eq c A s v (index q) i,Host.offset_eq c A s v (index q) j] at eq
    exact (Host.writer_unique c A s v _ _ i j
      ((Host.active_eq c A s v (index q) i).mp hi) ((Host.active_eq c A s v (index q) j).mp hj) eq).2
  obtain ⟨final,run,read,frame⟩ := EpilogueResult.correct c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)
    (stage adapter input q) (by rw [c.blockN];decide) bounds.advance injective
  refine ⟨final,run,?_⟩
  have live : c.active (index q) (c.lane (row q) (dim q)) := by
    unfold Host.Call.active
    rw [index_launch]
    simpa [Formula.queryRow,Host.Call.lane,c.blockM,Nat.mul_comm,Nat.div_add_mod,row,batch] using q.property
  have value := read (c.lane (row q) (dim q)) ((Host.active_eq c A s v _ _).mpr live)
  have off : c.offset (index q) (c.lane (row q) (dim q))=offset q := by
    unfold Host.Call.offset
    rw [index_launch]
    simp [Formula.queryRow,Host.Call.lane,c.blockM,Nat.mul_comm,Nat.div_add_mod,offset]
  change final.readMem c.regions.out (RealIO.write c.layout (Program.launch (s.withGridIndex (index q)))
      (Program.inputs (s.withGridIndex (index q)) c.regions) (c.lane (row q) (dim q)))=_ at value
  rw [Host.offset_eq c A s v (index q),off] at value
  rw [value]
  change Formula.expected c.layout (c.launchAt (index q)) (Program.inputs s c.regions)
    (Host.scale c.layout.D) (row q%128) (dim q).val=_
  rw [index_launch]
  exact Logical.expected_eq c A s (logical input) v (adapter.reads input) _ _ _ _
    q.val.1.isLt q.val.2.1.isLt q.property (dim q).isLt

theorem source_target (g : Geometry c tokens) (adapter : Adapter c tokens)
    (input : Recurrence.Inputs ℝ) (q : Coord c) : source adapter input q=target c input q := by
  obtain ⟨final,run,value⟩ := source_success g adapter input q
  unfold source
  rw [run]
  exact value

#axiomsClean source_success
#axiomsClean source_target
#specNonCircular target avoiding [stepStmts,source,execute]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.Mathematical
