import bench.repairs.tritonbench_g.triton_matmul.Proofs.OutputBytes
import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteGrid
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.ByteExecution
open VeriTile.Triton FP.Structural
open FPValue (Storage stored CIndex BM BN DotAcc dtype)
open TensorBinding OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 1000000

abbrev Machine := State Word × Read

def step (r : Request) (c : Tensor) (pid : Nat) (values : CIndex r.parameters → Word)
    (t : Machine) (i : CIndex r.parameters) : Machine :=
  if Index.storeMask r pid i.1.val i.2.1.val then
    (t.1.write r.C (FiniteExecution.outputOffset r pid i).toNat (stored r.storage (values i)),
      store t.2 r.storage (Index.cPointer r c.metadata.dataPtr.toNat pid i.1.val i.2.1.val).toNat (values i))
  else t

def logicalStep (r : Request) (pid : Nat) (values : CIndex r.parameters → Word)
    (s : State Word) (i : CIndex r.parameters) : State Word :=
  if Index.storeMask r pid i.1.val i.2.1.val then
    s.write r.C (FiniteExecution.outputOffset r pid i).toNat (stored r.storage (values i)) else s

theorem offset_bound (r : Request) (basis : State Word) (t : TensorAPI r basis)
    (pid : Nat) (hp : pid<programs r) (hl : programs r<2^31) (i : CIndex r.parameters) :
    (FiniteExecution.outputOffset r pid i).toNat<r.rows*r.columns := by
  have nonempty := nonempty_program r pid hp
  rw [FiniteExecution.output_offset r basis t basis pid hp hl i,candidate_store_offset]
  exact safeC_bound r _ _ nonempty.1 nonempty.2

theorem store_address (r : Request) (c : Tensor) (basis : State Word) (t : TensorAPI r basis)
    (pid : Nat) (hp : pid<programs r) (hl : programs r<2^31)
    (base : t.baseC=c.metadata.dataPtr.toNat) (i : CIndex r.parameters) :
    (Index.cPointer r c.metadata.dataPtr.toNat pid i.1.val i.2.1.val).toNat=
      c.metadata.dataPtr.toNat+width r.storage*(FiniteExecution.outputOffset r pid i).toNat := by
  rw [←base,(Index.typed_output r basis t pid i hp hl).1,
    FiniteExecution.output_offset r basis t basis pid hp hl i]
  have ew : elementBytes r.storage=width r.storage := by cases r.storage <;> rfl
  rw [ew]

theorem fold_render (r : Request) (c : Tensor) (basis : State Word) (t : TensorAPI r basis)
    (pid : Nat) (hp : pid<programs r) (hl : programs r<2^31) (hc : r.C=outputC)
    (size : r.rows*r.columns=c.metadata.numel) (base : t.baseC=c.metadata.dataPtr.toNat) (values : CIndex r.parameters → Word)
    (read : Read) (items : List (CIndex r.parameters)) (s : State Word) :
    items.foldl (step r c pid values) (s,render read r.storage c s)=
      let final := items.foldl (logicalStep r pid values) s
      (final,render read r.storage c final) := by
  induction items generalizing s with
  | nil => rfl
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases active : Index.storeMask r pid i.1.val i.2.1.val=Bool.true
    · have hb : (FiniteExecution.outputOffset r pid i).toNat<c.metadata.numel := by
        rw [←size]; exact offset_bound r basis t pid hp hl i
      simp only [step,logicalStep,if_pos active,store_address r c basis t pid hp hl base i,hc,←render_write read r.storage c s _ hb]
      exact ih _
    · simp only [step,logicalStep,if_neg active]
      exact ih s

def program (M : Algebra Word) (D : DotAcc Word) (r : Request) (c : Tensor)
    (pid : Nat) (t : Machine) : Option Machine :=
  (FiniteExecution.loop M D r t.1 pid (Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage)).toNat
    (Index.word 0) (fun _ => M.literal (some .fp32) .real 0)).map fun carry =>
    (TileShape.allIndices [BM r.parameters,BN r.parameters]).foldl
      (step r c pid (fun i => M.cast none .real (dtype r.storage) (carry i))) (instanceState r pid t.1,t.2)

theorem program_render (M : Algebra Word) (D : DotAcc Word) (r : Request) (c : Tensor)
    (basis : State Word) (t : TensorAPI r basis) (pid : Nat) (hp : pid<programs r) (hl : programs r<2^31)
    (hc : r.C=outputC) (size : r.rows*r.columns=c.metadata.numel) (base : t.baseC=c.metadata.dataPtr.toNat) (read : Read) (s : State Word) :
    program M D r c pid (s,render read r.storage c s)=
      (FiniteExecution.program M D r pid s).map (fun final => (final,render read r.storage c final)) := by
  simp only [program,FiniteExecution.program,Option.map_map]
  apply Option.map_congr
  intro carry _
  exact fold_render r c basis t pid hp hl hc size base _ read _ (instanceState r pid s)

def grid (M : Algebra Word) (D : DotAcc Word) (r : Request) (c : Tensor) : List Nat → Machine → Option Machine
  | [],t => some t
  | pid::tail,t => (program M D r c pid t).bind (grid M D r c tail)

theorem grid_render (M : Algebra Word) (D : DotAcc Word) (r : Request) (c : Tensor)
    (basis : State Word) (t : TensorAPI r basis) (hl : programs r<2^31)
    (hc : r.C=outputC) (size : r.rows*r.columns=c.metadata.numel) (base : t.baseC=c.metadata.dataPtr.toNat) (read : Read)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<programs r) (s : State Word) :
    grid M D r c ids (s,render read r.storage c s)=
      (FiniteGrid.grid M D r ids s).map (fun final => (final,render read r.storage c final)) := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    simp only [grid,FiniteGrid.grid,program_render M D r c basis t pid hp hl hc size base read s]
    cases he : FiniteExecution.program M D r pid s with
    | none => rfl
    | some current =>
      simpa only [Option.map_some,Option.bind_some] using ih (fun q hq => valid q (by simp [hq])) current

#axiomsClean store_address
#axiomsClean offset_bound
#axiomsClean program_render
#axiomsClean grid_render
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.ByteExecution
