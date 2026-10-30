import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Program

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

def atIndex {g : Grid} (s : State α) (idx : GridIndex g) : State α :=
  {s with pids:=idx.toPids,numPids:=g.toNumPids}

def runGrid (c : Host.Call) (o : FPBackend.Ops α) : List (GridIndex c.grid) → State α → Option (State α)
  | [],s => some s
  | idx::rest,s => do
      let t ← exec (engine o.algebra) (program c.layout c.regions .fp32 (Host.scale c.layout.D)) (atIndex s idx)
      runGrid c o rest t

def allPrograms (c : Host.Call) : List (GridIndex c.grid) := Finset.univ.toList
theorem allPrograms_complete (c : Host.Call) (idx : GridIndex c.grid) : idx∈allPrograms c := by simp [allPrograms]

def tile (c : Host.Call) (o : FPBackend.Ops α) (idx : GridIndex c.grid) (s : State α) : State α :=
  let launch := atIndex s idx
  scatter c.layout c.regions launch launch
    (output c.layout o (Host.scale c.layout.D) (inputs c.layout c.regions launch))

/-- The source-independent tile recurrence and its logical scatter for each
grid coordinate, without executing or inspecting the source quotation. -/
def grid (c : Host.Call) (o : FPBackend.Ops α) : List (GridIndex c.grid) → State α → State α
  | [],s => s
  | idx::rest,s => grid c o rest (tile c o idx s)

theorem tile_mem_congr (c : Host.Call) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (s t : State α) (memory : s.mem=t.mem) : (tile c o idx s).mem=(tile c o idx t).mem := by
  have inp : inputs c.layout c.regions (atIndex s idx)=inputs c.layout c.regions (atIndex t idx) := by
    simp only [inputs,atIndex,memory]
  dsimp only [tile]
  rw [inp]
  classical
  unfold scatter
  simp only [active,offset,inputs,atIndex,memory]
  apply masked_fold_mem_congr
  rfl

theorem grid_mem_congr (c : Host.Call) (o : FPBackend.Ops α) (order : List (GridIndex c.grid))
    (s t : State α) (memory : s.mem=t.mem) : (grid c o order s).mem=(grid c o order t).mem := by
  induction order generalizing s t with
  | nil => exact memory
  | cons idx rest ih => exact ih _ _ (tile_mem_congr c o idx s t memory)

/-- Every complete source CTA executes, threading actual memory, and the
result is the separately defined recurrence grid for any supplied ordering. -/
theorem whole (c : Host.Call) (o : FPBackend.Ops α) (order : List (GridIndex c.grid)) (s : State α) :
    ∃ t,runGrid c o order s=some t ∧ t.mem=(grid c o order s).mem := by
  have bn : 0<c.layout.BN := by rw [c.blockN];decide
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,he,hm⟩ := exec_complete c.layout c.regions (Host.scale c.layout.D) o bn (atIndex s idx)
    obtain ⟨final,hfinal,hfm⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · simp only [runGrid,he,Option.bind_some]
      exact hfinal
    · exact hfm.trans (grid_mem_congr c o rest mid (tile c o idx s) hm)

#specNonCircular grid avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,program,parts,runGrid]
#axiomsClean whole
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
