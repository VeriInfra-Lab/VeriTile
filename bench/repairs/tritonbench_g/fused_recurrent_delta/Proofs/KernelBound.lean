import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelTensor
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelExpected
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelGrid
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBound
open VeriTile.Triton FP.Structural KernelProgram KernelTensor KernelNativeCarrier
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

/-- Per-object pointer attributes are checked against the same Tensor
arguments. Optional None slots carry no pointer alignment obligation. -/
def PointerABI (platform : KernelCompiler.Platform) (binding : KernelCompiler.Binding platform kind s f)
    (b : Buffers kind s f) : Prop :=
  List.Forall₂ (fun divisor address => 0<divisor ∧ address%divisor=0)
    binding.object.signature.pointerDivisibility
    (((KernelCompiler.pointerSlots kind f).filterMap id).map fun r => (b.tensor r).dataPtr.toNat)

theorem coordinate_bound (b : Buffers kind s f) (a : Addr) (role : a.region∈KernelContract.roles kind f)
    (valid : Geometry.Valid s f a) : Geometry.offset s f a<(b.tensor (region a.region)).metadata.numel := by
  rw [b.count a.region role]
  exact Geometry.offset_bound s f a valid
theorem coordinate_i32 (b : Buffers kind s f) (a : Addr) (valid : Geometry.Valid s f a) :
    (BitVec.ofInt 32 (Geometry.offset s f a)).toInt=Geometry.offset s f a :=
  Index.active_i32_exact s f b.admitted.domain.toSafe a valid

def InputFrame {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (b : Buffers kind s f) (p : Program kind s) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) : Prop :=
  ∀ r,r∈KernelContract.roles kind f → r∉KernelContract.outputRoles kind f →
    ∀ i,i<(b.tensor (region r)).metadata.numel →
      api.read (Shared.commit api (placement b p) observe v initial) ((b.view (region r)).address i)=
        api.read initial ((b.view (region r)).address i)
theorem input_frame {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (b : Buffers kind s f) (p : Program kind s) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) : InputFrame api b p observe v initial := by
  intro r hr outside i hi
  apply Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  have active := (cells_coverage kind s f p a.1 a.2).mp ha
  obtain ⟨q,hq,eq,_⟩ := KernelGeometry.output_bounds kind s f p a.1 a.2 active
  have different : q≠r := by intro same; subst q; exact outside hq
  change Memory.Apart ((b.view a.1).address a.2) ((b.view (region r)).address i)
  rw [←eq]
  exact distinct_roles_apart b q r (KernelGeometry.outputs_are_arguments kind f q hq) hr different
    a.2 i (by rw [eq]; exact output_bound b p a.1 a.2 active) hi

def nativeRead {Store Byte : Type} (api : Memory.API Word Store Byte) (initial : Store)
    (r : RegionName) (i address : Nat) : Node := .atom (.input r i) (api.read initial address)

theorem native_tensor {Store Byte : Type} (api : Memory.API Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (kind : Kind) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform kind s f) (arg : KernelCompiler.ScaleArgument)
    (scale : KernelCompiler.ScaleAdmitted platform binding arg)
    (b : Buffers kind s f) (abi : PointerABI platform binding b) (p : Program kind s) (u : State Node)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    PointerABI platform binding b ∧ KernelCompiler.ScaleAdmitted platform binding arg ∧ ∃ v,
      KernelCompiler.Facts platform binding (KernelCompiler.hostScale platform s arg) p
        (physical b (nativeRead api initial) u) v ∧
      Shared.TensorImage api (placement b p)
        (fun _ _ => observe (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)) v initial ∧
      InputFrame api b p
        (fun _ _ => observe (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)) v initial ∧
      (∀ r off word,KernelCompiler.Expected platform binding (KernelCompiler.hostScale platform s arg) p
        (logical b (nativeRead api initial) u) r off word →
        (VeriTile.Torch.FP32.logicalValue (fun a => (⟨api.read (Shared.commit api (placement b p)
          (fun _ _ => observe (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)) v initial) a⟩ :
            VeriTile.Torch.FP32.Word)) (b.tensor r) (b.laws r) off).bits=word) := by
  obtain ⟨v,facts,values⟩ := KernelCompiler.complete platform kind s f binding
    (KernelCompiler.hostScale platform s arg) p (physical b (nativeRead api initial) u) hv hk hh
  rw [physical_eq_logical] at values
  refine ⟨abi,scale,v,facts,Shared.complete api (placement b p) _ v initial,
    input_frame api b p _ v initial,?_⟩
  intro r off word expected
  rw [native_output api b p (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)
    v initial r off (KernelExpected.written platform binding _ p _ r off word expected)]
  exact values r off word expected

#axiomsClean coordinate_bound
#axiomsClean coordinate_i32
#axiomsClean input_frame
#axiomsClean native_tensor
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBound
