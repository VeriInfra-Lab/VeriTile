import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelBound

/-! Optional actual-Tensor applications, admission correspondences and physical
binding facts. Kernel mathematics and equivalence stay in the primary entries. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.TensorBridge
open VeriTile.Triton FP.Structural KernelProgram KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
variable {α β Store Byte : Type} [Inhabited α]

section
open KernelBound
theorem chunk_gate_recurrence_tensor_real (api : Memory.API ℝ Store Byte) (initial : Store)
    (b : Buffers c s kind seed) (currentDevice : Nat) (accepted : KernelLaunch.check b currentDevice=Bool.true)
    (nonempty : 0<c.N) (p : Program c) (out : Outputs b p) (isolated : Isolated b p out) (u : State ℝ)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    KernelLaunch.Admitted b currentDevice ∧ ∃ v,
      KernelExecution.Facts KernelRealReference.algebra kind c s seed p (physical b (api.read initial) u) v ∧
      KernelReal.Outputs kind c s seed p (physical b (api.read initial) u) v ∧
      Shared.TensorImage api out.toPlacement (fun r _ => rawCell kind r) v initial ∧
      InputFrame api b p out (fun r _ => rawCell kind r) v initial ∧
      TensorResults api b p out (fun r _ => rawCell kind r) v initial
        (KernelReal.expected kind c s seed p (physical b (api.read initial) u)) :=
  KernelBound.real_tensor api initial b currentDevice accepted nonempty p out isolated u hh hk hv
end

section
open KernelBound
theorem chunk_gate_recurrence_tensor_unwritten (api : Memory.API β Store Byte) (b : Buffers c s kind seed)
    (p : Program c) (out : Outputs b p) (observe : RegionName → Nat → Cell α → β)
    (v : State α) (initial : Store) (r : RegionName)
    (hr : r∈KernelExecution.outputRoles kind) (i : Nat) (hi : i<(b.tensor r).metadata.numel)
    (outside : ¬KernelMemory.footprint kind c p r i) :
    api.read (Shared.commit api out.toPlacement observe v initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))=
      api.read initial ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)) :=
  KernelBound.unwritten_output_frame api b p out observe v initial r hr i hi outside
end

section
open KernelLaunch
theorem chunk_gate_recurrence_check_iff (b : Buffers c s kind seed) (currentDevice : Nat) :
    check b currentDevice=Bool.true ↔ Admitted b currentDevice :=
  KernelLaunch.check_iff b currentDevice
end

section
open KernelLaunch
theorem chunk_gate_recurrence_source_ranges (b : Buffers c s kind seed) (current : Nat) (h : Admitted b current)
    (p : Program c) (e : Fin 64) (j : Fin 16) (count : Fin c.N)
    (x : Machine.Expr) (member : x∈Machine.sourceTrees c (strides s) p.head p.key p.value e.val j.val count.val) :
    x.Valid ∧ x.word.toInt=x.eval :=
  KernelLaunch.expressions b current h p e j count x member
end

section
open KernelLaunch
theorem chunk_gate_recurrence_formed_pointers (b : Buffers c s kind seed) (current : Nat) (h : Admitted b current)
    (p : Program c) (e : Fin 64) (j : Fin 16) (count : Fin c.N) (r : RegionName) (hr : r∈roles kind seed)
    (x : Machine.Expr) (member : x∈Machine.pointerExprs c (strides s) seed
      p.head p.key p.value e.val j.val count.val (role kind r)) :
    x.Valid ∧ x.word.toInt=x.eval ∧
      0≤((b.tensor r).dataPtr.toNat:Int)+4*x.eval ∧ ((b.tensor r).dataPtr.toNat:Int)+4*x.eval<2^64 :=
  KernelLaunch.formed b current h p e j count r hr x member
end

section
open KernelBounds
theorem chunk_gate_recurrence_word_bounds (b : Buffers c s kind seed) (r : RegionName) (off : Nat) (hi : off<extent b r) :
    (b.tensor r).storage.base≤KernelTensor.address b r off ∧
    KernelTensor.address b r off+4≤(b.tensor r).storage.base+4*(b.tensor r).storage.elements ∧
    KernelTensor.address b r off+4≤2^64 :=
  KernelBounds.physical_word b r off hi
end

section
open KernelBound
theorem chunk_gate_recurrence_tensor_equiv (api : Memory.API KernelCarrier.Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (binding : KernelCompiler.Binding platform kind c s seed)
    (b : Buffers c s kind seed) (currentDevice : Nat) (accepted : KernelLaunch.check b currentDevice=Bool.true)
    (abi : Pointers platform binding b) (nonempty : 0<c.N) (p : Program c)
    (out : Outputs b p) (isolated : Isolated b p out) (u : State KernelCarrier.Node)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    KernelLaunch.Admitted b currentDevice ∧ Pointers platform binding b ∧ ∃ v,
      KernelCompiler.Facts platform binding p (physical b (fun a => KernelCarrier.atom (api.read initial a)) u) v ∧
      KernelNative.Outputs (platform.instructions binding.object.primitives) (KernelCompiler.plan platform binding)
        kind c s seed p (physical b (fun a => KernelCarrier.atom (api.read initial a)) u) v ∧
      Shared.TensorImage api out.toPlacement (observe kind) v initial ∧
      InputFrame api b p out (observe kind) v initial ∧
      TensorResults api b p out (observe kind) v initial
        (KernelLogical.expected (platform.instructions binding.object.primitives) (KernelCompiler.plan platform binding)
          b (api.read initial) p) :=
  KernelBound.native_tensor api initial platform binding b currentDevice accepted abi nonempty p out isolated u hh hk hv
end

section
open KernelLogical KernelCarrier
theorem chunk_gate_recurrence_logical_inputs (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (b : Buffers c s kind seed) (read : Nat → Word) (p : Program c) (u : State Node)
    (r : RegionName) (t : Fin c.N) (i : Lane) :
    KernelNative.expected isa plan kind c s seed p (physical b (fun a => atom (read a)) u) r t.val i=
      expected isa plan b read p r t.val i :=
  KernelLogical.expected_eq isa plan b read p u r t i
end

#axiomsClean chunk_gate_recurrence_tensor_real
#axiomsClean chunk_gate_recurrence_tensor_unwritten
#axiomsClean chunk_gate_recurrence_check_iff
#axiomsClean chunk_gate_recurrence_source_ranges
#axiomsClean chunk_gate_recurrence_formed_pointers
#axiomsClean chunk_gate_recurrence_word_bounds
#axiomsClean chunk_gate_recurrence_tensor_equiv
#axiomsClean chunk_gate_recurrence_logical_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.TensorBridge
