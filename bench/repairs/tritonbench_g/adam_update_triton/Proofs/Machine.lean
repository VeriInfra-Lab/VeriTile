import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Private
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Native
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Real
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Machine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton (Channel)
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune (Config)
noncomputable section
set_option maxHeartbeats 6000000

/-- The stored carrier and source-expression carrier may differ. In the native
case source expressions are literal typed graphs and stored values are raw
FP32 words. This avoids pretending that raw-byte loads/stores preserve ASTs. -/
structure Model (α τ : Type) [Inhabited τ] where
  semantics : Tile.Semantics τ
  arguments : Scalars τ
  input : α → τ
  observe : τ → α
  input_observe : ∀ value,observe (input value)=value

def scalar {α τ : Type} [Inhabited τ] (model : Model α τ)
    (output : Output) (p g m : α) : α :=
  model.observe (Tile.scalar model.semantics model.arguments output (model.input p) (model.input g) (model.input m))

def nativeModel (native : Native.Instructions α) (params : Scalars α) :
    Model α (VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.Term α) :=
  ⟨Native.semantics native,Native.arguments params,.input,Native.interpret native,fun _ => rfl⟩

def realModel (params : Scalars ℝ) : Model ℝ ℝ :=
  ⟨Real.semantics,params,id,id,fun _ => rfl⟩

theorem native_scalar (native : Native.Instructions α) (params : Scalars α) (output : Output) (p g m : α) :
    scalar (nativeModel native params) output p g m=Native.reference native params output p g m :=
  Native.lowering native params output p g m

theorem real_scalar (params : Scalars ℝ) (output : Output) (p g m : ℝ) :
    scalar (realModel params) output p g m=Math.target params output p g m :=
  Real.scalar_target params output p g m

def channels {α τ : Type} [Inhabited α] [Inhabited τ] (model : Model α τ) : Channel → α → α → α → α
  | .c,_,_,_ => default
  | .da,p,m,g => scalar model .parameter p g m
  | .db,p,m,g => scalar model .momentum p g m

def run {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (indices : List Nat) (s : M) : M :=
  VeriTile.Bench.Repairs.SwigluTriton.Physical.run ops (channels model) Bool.true p.flat indices s

theorem values {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (valid : Private.Valid p) (indices : List Nat)
    (unique : indices.Nodup) (bounded : ∀ i∈indices,i<p.total) (s : M) (output : Output)
    (i : Nat) (member : i∈indices) :
    ops.read (run ops model p indices s) ((p.allocation (Private.outputRole output)).key i)=
      scalar model output (ops.read s ((p.allocation .a).key i))
        (ops.read s ((p.allocation .third).key i)) (ops.read s ((p.allocation .b).key i)) := by
  have h := VeriTile.Bench.Repairs.SwigluTriton.Physical.run_values ops (channels model) Bool.true
    p.flat valid indices unique bounded s (Private.channel output) (by cases output <;> decide)
    i 0 member (by change 0<1; decide)
  cases output <;> simpa [run,Private.Layout.flat,Private.channel,Private.outputRole,channels,
    VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
    VeriTile.Bench.Repairs.SwigluTriton.Physical.value,
    VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole] using h

theorem byte_frame {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (indices : List Nat) (s : M) (owner address : Nat)
    (outside : ∀ i∈indices,∀ output,¬((p.allocation (Private.outputRole output)).key i).covers owner address) :
    ops.observe (run ops model p indices s) owner address=ops.observe s owner address := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Physical.run_bytes
  intro i member ch on col active
  have zero : col=0 := by change col<1 at active; omega
  subst col
  cases ch
  · simp [VeriTile.Bench.Repairs.SwigluTriton.enabled] at on
  · simpa [Private.Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,Private.outputRole] using outside i member .parameter
  · simpa [Private.Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,Private.outputRole] using outside i member .momentum

theorem external_frame {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (indices : List Nat) (s : M) (key : Key)
    (outside : ∀ i∈indices,∀ output,key.disjoint ((p.allocation (Private.outputRole output)).key i)) :
    ops.read (run ops model p indices s) key=ops.read s key := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Physical.run_frame
  intro i member ch on col active
  have zero : col=0 := by change col<1 at active; omega
  subst col
  cases ch
  · simp [VeriTile.Bench.Repairs.SwigluTriton.enabled] at on
  · simpa [Private.Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,Private.outputRole] using outside i member .parameter
  · simpa [Private.Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key,
      VeriTile.Bench.Repairs.SwigluTriton.Physical.outputRole,Private.outputRole] using outside i member .momentum

theorem read_frame {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (valid : Private.Valid p) (indices : List Nat)
    (bounded : ∀ j∈indices,j<p.total) (s : M) (role : Role) (i : Nat) (active : i<p.total)
    (outside : ∀ j∈indices,∀ output,role≠Private.outputRole output ∨ i≠j) :
    ops.read (run ops model p indices s) ((p.allocation role).key i)=ops.read s ((p.allocation role).key i) := by
  apply external_frame
  intro j member output
  have different : p.flat.key role i 0≠p.flat.key (Private.outputRole output) j 0 := by
    intro equal
    have unique := VeriTile.Bench.Repairs.SwigluTriton.Physical.keys_unique p.flat valid
      role (Private.outputRole output) i 0 j 0 active (by change 0<1; decide)
      (bounded j member) (by change 0<1; decide) equal
    rcases outside j member output with different | different
    · exact different unique.1
    · exact different unique.2.1
  simpa [Private.Layout.flat,VeriTile.Bench.Repairs.SwigluTriton.Physical.Layout.key] using
    VeriTile.Bench.Repairs.SwigluTriton.Physical.keys_disjoint p.flat valid role (Private.outputRole output)
      i 0 j 0 active (by change 0<1; decide) (bounded j member) (by change 0<1; decide) different

def state {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (first : Nat) (s : M) : State τ where
  mem := fun region offset => Cell.mk .real (model.input
    (if region="P" then ops.read s ((p.allocation .a).key (first+offset))
     else if region="M" then ops.read s ((p.allocation .b).key (first+offset))
     else if region="G" then ops.read s ((p.allocation .third).key (first+offset))
     else default))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem state_read {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (first : Nat) (s : M) (role : Role) (i : Nat) :
    model.observe (((state ops model p first s).mem (Private.region role) i).read .real)=
      ops.read s ((p.allocation role).key (first+i)) := by
  cases role <;> simp [state,Private.region,model.input_observe]

structure SourceStep {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (chunk : Index.Chunk) (config : Config) (schedule : List Nat) (before after : M) : Prop where
  schedule_complete : schedule.Perm (List.range (Index.grid chunk.count config))
  physical : after=run ops model p chunk.indices before
  source : ∃ final,Grid.runPrograms model.semantics model.arguments p.total chunk.count config.block schedule
      (state ops model p chunk.first before)=some final ∧
    ∀ role i,chunk.first+i<p.total →
      model.observe ((final.mem (Private.region role) i).read .real)=
        ops.read after ((p.allocation role).key (chunk.first+i))

theorem step {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (valid : Private.Valid p) (chunk : Index.Chunk)
    (fits : chunk.first+chunk.count≤p.total) (config : Config) (schedule : List Nat)
    (complete : schedule.Perm (List.range (Index.grid chunk.count config))) (s : M) :
    SourceStep ops model p chunk config schedule s (run ops model p chunk.indices s) := by
  obtain ⟨final,ran,sourceValues,sourceFrame⟩ := Grid.whole model.semantics model.arguments p.total chunk.count config.block
    (by cases config <;> decide) schedule complete (state ops model p chunk.first s)
  have bounded : ∀ j∈chunk.indices,j<p.total := by
    intro j member
    have := List.mem_range'_1.mp member
    omega
  refine ⟨complete,rfl,final,ran,?_⟩
  intro role i active
  by_cases inside : i<chunk.count
  · have value (output : Output) :
        model.observe ((final.mem (Tile.outputRegion output) i).read .real)=
          ops.read (run ops model p chunk.indices s) ((p.allocation (Private.outputRole output)).key (chunk.first+i)) := by
      rw [sourceValues output i inside,values ops model p valid chunk.indices List.nodup_range'
        bounded s output (chunk.first+i) (List.mem_range'_1.mpr (by omega))]
      simp [state,scalar]
    cases role
    · exact value .parameter
    · exact value .momentum
    · rw [sourceFrame (Private.region .third) i (Or.inl (by decide)),state_read,
        read_frame ops model p valid chunk.indices bounded s .third (chunk.first+i) active]
      intro j member output; left; cases output <;> decide
  · rw [sourceFrame (Private.region role) i (Or.inr (by omega)),state_read,
      read_frame ops model p valid chunk.indices bounded s role (chunk.first+i) active]
    intro j member output
    right
    have := List.mem_range'_1.mp member
    omega

def chunks {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) : List Index.Chunk → M → M
  | [],s => s
  | chunk::rest,s => chunks ops model p rest (run ops model p chunk.indices s)

theorem run_append {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (a b : List Nat) (s : M) :
    run ops model p (a++b) s=run ops model p b (run ops model p a s) := by
  induction a generalizing s with
  | nil => rfl
  | cons i rest ih => exact ih _

theorem chunks_flatten {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (parts : List Index.Chunk) (s : M) :
    chunks ops model p parts s=run ops model p (parts.flatMap Index.Chunk.indices) s := by
  induction parts generalizing s with
  | nil => rfl
  | cons head rest ih => rw [chunks,ih,List.flatMap_cons,run_append]

def whole {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (s : M) : M := chunks ops model p (Index.chunks 0 p.total) s

theorem whole_values {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Model α τ)
    (p : Private.Layout) (valid : Private.Valid p) (s : M) (output : Output) (i : Nat) (active : i<p.total) :
    ops.read (whole ops model p s) ((p.allocation (Private.outputRole output)).key i)=
      scalar model output (ops.read s ((p.allocation .a).key i))
        (ops.read s ((p.allocation .third).key i)) (ops.read s ((p.allocation .b).key i)) := by
  rw [whole,chunks_flatten,Index.schedule]
  exact values ops model p valid _ List.nodup_range (fun j hj => List.mem_range.mp hj) s output i (List.mem_range.mpr active)

#axiomsClean step
#axiomsClean native_scalar
#axiomsClean whole_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Machine
