import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.IOBounds

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open IOData
noncomputable section
open Classical
set_option maxHeartbeats 3000000
variable {α : Type}

abbrev Memory (α : Type) := RegionName → Nat → α

def arguments (c : Call) (r : Regions) (idx : GridIndex c.grid) (memory : Memory α) : FPReference.Inputs α where
  query := memory r.query
  key := memory r.key
  value := memory r.value
  head := (c.launch idx).head
  block := (c.launch idx).block
  start := c.start (c.launch idx).batch
  length := c.length (c.launch idx).batch

def SameReads (c : Call) (r : Regions) (left right : Memory α) : Prop :=
  ∀ region offset,ReadHit c r region offset → left region offset=right region offset

def output (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid) (memory : Memory α) :=
  FPReference.output c.layout o c.scale (arguments c r idx memory)

/-- Source-independent last-writer update of a logical payload table. This
retains collisions and in-place Q/K/V aliases in the original native domain. -/
def scatter (c : Call) (r : Regions) (idx : GridIndex c.grid)
    (values : Values α .real [c.layout.BM,c.layout.D]) (memory : Memory α) : Memory α :=
  (TileShape.allIndices [c.layout.BM,c.layout.D]).foldl (fun current lane =>
    if FPReference.laneActive c idx lane then
      fun region offset => if region=r.out ∧ offset=FPReference.laneOffset c idx lane then values lane else current region offset
    else current) memory

def step (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid) (memory : Memory α) : Memory α :=
  scatter c r idx (output c r o idx memory) memory

def grid (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (memory : Memory α) : Memory α :=
  order.foldl (fun current idx => step c r o idx current) memory

#specNonCircular grid avoiding [FPReference.grid,FPReference.runGrid,FP.Structural.exec,FP.Structural.run,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
