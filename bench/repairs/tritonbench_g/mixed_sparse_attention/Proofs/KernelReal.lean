import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RawCorrect
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelTensor
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.RealReturned

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelReal
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
set_option maxHeartbeats 6000000

def initial (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (integer : ℝ → Int) : State Real.R :=
  KernelTensor.initial c request layout read 0 some integer
def values (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (r : RegionName) (i : Nat) : ℝ :=
  match layout.route r with | some role=>readRole c request read 0 role i | none=>0
theorem reads (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (integer : ℝ → Int) :
    Real.Reads c (initial c request layout read integer) (values c request layout read) := by
  have hq : layout.route c.q=some Role.query := layout.inputs .query (by simp [inputRoles])
  have hk : layout.route c.k=some Role.key := layout.inputs .key (by simp [inputRoles])
  have hv : layout.route c.v=some Role.value := layout.inputs .value (by simp [inputRoles])
  intro r hr i
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl
  · simpa [initial,values,hq,Role.floating,Role.region,ofFloat] using
      (input_cell (request:=request) layout .query (by simp [inputRoles]) read 0 some integer i)
  · simpa [initial,values,hk,Role.floating,Role.region,ofFloat] using
      (input_cell (request:=request) layout .key (by simp [inputRoles]) read 0 some integer i)
  · simpa [initial,values,hv,Role.floating,Role.region,ofFloat] using
      (input_cell (request:=request) layout .value (by simp [inputRoles]) read 0 some integer i)

def target (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (integer : ℝ → Int) (q : Raw.Coord c) : ℝ :=
  Real.sparseExpected c (initial c request layout read integer) (values c request layout read)
    (Raw.hz q) (Raw.row q) (Raw.dim q)
def observe (cell : Cell Real.R) : ℝ := (cell.read .real).unbotD 0
structure Result (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (integer : ℝ → Int) (final : State Real.R) : Prop where
  output : ∀ q : Raw.Coord c,live c (initial c request layout read integer) q →
    final.mem c.out (Raw.offset q)=.mk .real (some (target c request layout read integer q))
  frame : ∀ r i,¬Writes c (initial c request layout read integer) r i →
    final.mem r i=(initial c request layout read integer).mem r i

theorem source (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (integer : ℝ → Int)
    (accepted : Admission c request (observed c request read 0 integer))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) c (grid c) work
      (initial c request layout read integer)=some final ∧ Result c request layout read integer final := by
  let s:=initial c request layout read integer
  have hm:=Checked.metadata c _ s accepted.metadata (metadata_initial layout read 0 some integer)
  have hbm : 0<c.BM := by rw [accepted.schema.selected.2.1];decide
  have hbn : 0<c.BN := by rw [accepted.schema.selected.2.2];decide
  obtain ⟨final,he,output,frame⟩:=RawCorrect.kernel_real c accepted.separated accepted.contiguous s
    (values c request layout read) (reads c request layout read integer) hm.sequence hbm hbn work order
  exact ⟨final,he,⟨output,frame⟩⟩

theorem complete {Store Byte : Type} (api : Memory.API ℝ Store Byte)
    (c : Config) (request : Request) (layout : Reify.Layout c) (b : Buffers request)
    (integer : ℝ → Int) (store : Store)
    (accepted : Admission c request (observed c request (api.read store) 0 integer))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    let s:=initial c request layout (api.read store) integer
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) c (grid c) work s=some final ∧
      Result c request layout (api.read store) integer final ∧
      Shared.TensorImage api (placement b accepted s) (fun _ _=>observe) final store ∧
      (∀ q : Raw.Coord c,live c s q → api.read
        (Shared.commit api (placement b accepted s) (fun _ _=>observe) final store)
        ((b.tensor .output).storage.address ((b.tensor .output).physicalIndex (Raw.offset q)))=
          target c request layout (api.read store) integer q) := by
  dsimp only
  obtain ⟨final,he,hr⟩:=source c request layout (api.read store) integer accepted work order
  refine ⟨final,he,hr,Shared.complete _ _ _ _ _,?_⟩
  intro q hq
  have hw : Writes c (initial c request layout (api.read store) integer) c.out (Raw.offset q) :=
    ⟨rfl,q,hq,rfl⟩
  have h:=Shared.commit_output api (placement b accepted _) (fun _ _=>observe) final store c.out (Raw.offset q) hw
  rw [hr.output q hq] at h
  exact h

#axiomsClean reads
#axiomsClean source
#axiomsClean complete
#specNonCircular target avoiding [Source.finalValues,Source.update,FP.Structural.exec,Whole.runPrograms,Whole.invoke]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelReal
