import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.FPValues

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 3000000

/-- Unrestricted Real observations at actual Tensor logical addresses;
metadata/page words come from the same original physical INT32 heap. -/
def mathematicalInputs (r : Request) (heap : ByteMemory) (read : Nat → ℝ) : Formula.Inputs where
  q := logical read r.q
  k := logical read r.k
  v := logical read r.v
  start := indexValues r heap .start
  sequence := indexValues r heap .sequence
  prompt := indexValues r heap .prompt
  request := indexValues r heap .request
  pages := indexValues r heap .pages

/-- Independent finite-sentinel page-gather formula. Every rounded key lane
contributes to the denominator; masked key/value loads give zero payloads.
The five exact FP32 launch-scale values are retained, without an ideal-scale
or ideal-causal replacement. No native-to-Real equality is asserted. -/
def mathematical (r : Request) (heap : ByteMemory) (read : Nat → ℝ) (b head row dim : Nat) : ℝ :=
  let c := r.call
  let a := mathematicalInputs r heap read
  let l : Formula.Launch := ⟨row/c.layout.BM,b*c.layout.heads+head⟩
  (∑ j : Fin (Formula.count c.layout l a*c.layout.BN),
    Real.exp (Real.log 2*Formula.score c.layout l a (Host.scale c.layout.D) (row%c.layout.BM) j.val) *
      Formula.value c.layout l a j.val dim) /
  ∑ j : Fin (Formula.count c.layout l a*c.layout.BN),
    Real.exp (Real.log 2*Formula.score c.layout l a (Host.scale c.layout.D) (row%c.layout.BM) j.val)

/-- Real interpretation initialized from actual Tensor storage, not from any
desired attention result. The raw-pointer connection is proved by real_input. -/
def realInitial (r : Request) (heap : ByteMemory) (read : Nat → ℝ) : BlockState where
  mem := fun rg off => if rg=regions.start then ⟨.int,indexValues r heap .start off⟩
    else if rg=regions.sequence then ⟨.int,indexValues r heap .sequence off⟩
    else if rg=regions.prompt then ⟨.int,indexValues r heap .prompt off⟩
    else if rg=regions.request then ⟨.int,indexValues r heap .request off⟩
    else if rg=regions.pages then ⟨.int,indexValues r heap .pages off⟩
    else if rg=regions.query then .real (logical read r.q off)
    else if rg=regions.key then .real (logical read r.k off)
    else if rg=regions.value then .real (logical read r.v off)
    else .real (read (base r rg+4*off))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := r.call.grid.toNumPids

theorem real_inputs (r : Request) (heap : ByteMemory) (read : Nat → ℝ) :
    Program.inputs (realInitial r heap read) r.call.regions=mathematicalInputs r heap read := rfl

theorem real_input (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ)
    (role : InputRole) (i : Nat) (hi : i<(role.payload r).tensor.metadata.numel) :
    (realInitial r heap read).readMem role.region i=read (base r role.region+4*i) := by
  have eq := logical_direct read (role.payload r) (inputAPI r heap h role)
    (input_descriptor r heap h role).1 i hi
  cases role <;> exact eq

def real_observation (r : Request) (heap : ByteMemory) (read : Nat → ℝ) :
    Checked.MetadataObservation r.call (realInitial r heap read)
      (indexValues r heap .start) (indexValues r heap .sequence) (indexValues r heap .prompt)
      (indexValues r heap .request) (indexValues r heap .pages) where
  startWords := fun b => load32 heap (r.start.metadata.dataPtr.toNat+4*b)
  sequenceWords := fun b => load32 heap (r.sequence.metadata.dataPtr.toNat+4*b)
  promptWords := fun b => load32 heap (r.prompt.metadata.dataPtr.toNat+4*b)
  requestWords := fun b => load32 heap (r.request.metadata.dataPtr.toNat+4*b)
  pageWords := fun b => load32 heap (r.pages.metadata.dataPtr.toNat+4*b)
  startCPU := fun _ _ => rfl
  sequenceCPU := fun _ _ => rfl
  promptCPU := fun _ _ => rfl
  requestCPU := fun _ _ => rfl
  pagesCPU := fun _ => rfl
  startMemory := by intro b hb; rfl
  sequenceMemory := by intro b hb; rfl
  promptMemory := by intro b hb; rfl
  requestMemory := by intro b hb; rfl
  pagesMemory := by intro b; rfl

theorem real_valid (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ) :
    Host.Valid r.call (allocation r) (realInitial r heap read) :=
  Checked.valid r.call r.dimensions (allocation r) (realInitial r heap read)
    (indexValues r heap .start) (indexValues r heap .sequence) (indexValues r heap .prompt)
    (indexValues r heap .request) (indexValues r heap .pages)
    h.checks.2.1 h.checks.2.2.2 (real_observation r heap read) (labels r)

theorem expected_eq (r : Request) (heap : ByteMemory) (read : Nat → ℝ) (b head row dim : Nat) :
    r.call.logicalExpected (realInitial r heap read) b head row dim=mathematical r heap read b head row dim := by
  rw [Call.logicalExpected,real_inputs]
  simp only [mathematical,Formula.expected,Formula.numerator,Formula.denominator,Fin.is_lt,↓reduceIte]
  rfl

#specNonCircular mathematical avoiding [VeriTile.Triton.exec,FP.Structural.exec,execute,FPReference.output,FPReference.grid,nativeStores]
#specNonCircular realInitial avoiding [mathematical,mathematicalInputs]
#axiomsClean real_input
#axiomsClean real_valid
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
