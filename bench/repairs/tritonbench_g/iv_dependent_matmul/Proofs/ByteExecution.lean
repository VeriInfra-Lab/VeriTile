import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.OutputBytes

namespace VeriTile.Bench.Repairs.IvDependentMatmul.ByteExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 1000000

abbrev Machine := State Word × Read
def view (r : Request) (bc : Nat) : View := ⟨r.C,bc,r.rows*r.columns⟩

def step (r : Request) (bc pid : Nat) (values : CIndex r → Word)
    (t : Machine) (q : CIndex r) : Machine :=
  if Native.storeMask r pid q then
    (t.1.write r.C (((Native.cAt r bc pid q).toNat-bc)/4) (.mk .fp32 (values q)),
      store t.2 (Native.cAt r bc pid q).toNat (values q).bits)
  else t

def logicalStep (r : Request) (bc pid : Nat) (values : CIndex r → Word)
    (s : State Word) (q : CIndex r) : State Word :=
  if Native.storeMask r pid q then
    s.write r.C (((Native.cAt r bc pid q).toNat-bc)/4) (.mk .fp32 (values q)) else s

theorem fold_render (r : Request) (basis : State Word) (a : Allocation r basis)
    (d : Index64.Addressable r basis a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) (values : CIndex r → Word)
    (read : Read) (items : List (CIndex r)) (s : State Word) :
    items.foldl (step r a.baseC pid values) (s,render read (view r a.baseC) s)=
      let final := items.foldl (logicalStep r a.baseC pid values) s
      (final,render read (view r a.baseC) final) := by
  have cor := Native.correspondence r basis a d l
  induction items generalizing s with
  | nil => rfl
  | cons q tail ih =>
    simp only [List.foldl_cons]
    by_cases active : Native.storeMask r pid q=Bool.true
    · have mask : FPValue.active r.parameters (instanceState pid basis) q=Bool.true :=
        (cor.cMask pid hp q).symm.trans active
      have address := cor.activeC pid hp q mask
      have bound := store_bound r basis pid q mask
      simp only [step,logicalStep,if_pos active,address,Execution.decode_address]
      have rendered := render_write read (view r a.baseC) s _ bound (values q)
      change render read (view r a.baseC)
          (s.write r.C (outputOffset r.parameters (instanceState pid basis) q) (.mk .fp32 (values q)))=
        store (render read (view r a.baseC) s)
          (a.baseC+4*outputOffset r.parameters (instanceState pid basis) q) (values q).bits at rendered
      rw [←rendered]
      exact ih _
    · simp only [step,logicalStep,if_neg active]
      exact ih s

def program (m : Execution.Arithmetic Word r) (g : Graph) (ba bb bc pid : Nat) (t : Machine) : Machine :=
  let values := Execution.accumulated m g ba bb pid t.1
  (TileShape.allIndices [r.blockRows,r.blockColumns]).foldl
    (step r bc pid (fun q => m.widen (m.narrow (values q))))
    ({t.1 with pids:=fun ax => if ax=0 then pid else 0},t.2)

theorem program_render (m : Execution.Arithmetic Word r) (g : Graph)
    (basis : State Word) (a : Allocation r basis) (d : Index64.Addressable r basis a)
    (l : Index64.Launchable r) (pid : Nat) (hp : pid<programs r) (read : Read) (s : State Word) :
    program m g a.baseA a.baseB a.baseC pid (s,render read (view r a.baseC) s)=
      let final := Execution.program m g a.baseA a.baseB a.baseC pid s
      (final,render read (view r a.baseC) final) := by
  exact fold_render r basis a d l pid hp _ read _ _

def grid (m : Execution.Arithmetic Word r) (g : Graph) (ba bb bc : Nat) : List Nat → Machine → Machine
  | [],t => t
  | pid::tail,t => grid m g ba bb bc tail (program m g ba bb bc pid t)

theorem grid_render (m : Execution.Arithmetic Word r) (g : Graph)
    (basis : State Word) (a : Allocation r basis) (d : Index64.Addressable r basis a)
    (l : Index64.Launchable r) (read : Read) (ids : List Nat)
    (valid : ∀ pid∈ids,pid<programs r) (s : State Word) :
    grid m g a.baseA a.baseB a.baseC ids (s,render read (view r a.baseC) s)=
      let final := Execution.runPrograms m g a.baseA a.baseB a.baseC ids s
      (final,render read (view r a.baseC) final) := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [grid,Execution.runPrograms,program_render m g basis a d l pid (valid pid (by simp)) read s]
    exact ih (fun q hq => valid q (by simp [hq])) _

#axiomsClean fold_render
#axiomsClean program_render
#axiomsClean grid_render
end
end VeriTile.Bench.Repairs.IvDependentMatmul.ByteExecution
