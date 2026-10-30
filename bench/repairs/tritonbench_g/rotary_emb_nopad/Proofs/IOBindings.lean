import bench.repairs.tritonbench_g.rotary_emb_nopad.IO
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem u_pid0 (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) (s : State α) :
    ((uInvocation rest ops g p q).initial s).pids 0=p := by
  simp [uInvocation,Invocation.initial,bindRegisters]
@[simp] theorem u_pid1 (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) (s : State α) :
    ((uInvocation rest ops g p q).initial s).pids 1=q := by
  simp [uInvocation,Invocation.initial,bindRegisters]
@[simp] theorem c_pid0 (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) (s : State α) :
    ((cInvocation rest ops g p q).initial s).pids 0=p := by
  simp [cInvocation,Invocation.initial,bindRegisters]
@[simp] theorem c_pid1 (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) (s : State α) :
    ((cInvocation rest ops g p q).initial s).pids 1=q := by
  simp [cInvocation,Invocation.initial,bindRegisters]

theorem u_offset (g : Geometry) (p q : Nat) (s : State α) (hp : s.pids 0=p) (hq : s.pids 1=q)
    (key odd : Bool) (i : ULane g) :
    Typed.uOffset (Typed.uLayout g.request) s key odd i=uOffset g p q key odd i := by
  cases key <;> simp [Typed.uOffset,Typed.uToken,Typed.uHead,Typed.uDim,Typed.uLayout,
    uOffset,uToken,Request.heads,Typed.half_width,hp,hq,Geometry.request]
theorem u_coefficient (g : Geometry) (q : Nat) (s : State α) (hq : s.pids 1=q)
    (sine : Bool) (i : ULane g) :
    Typed.uCoefficient (Typed.uLayout g.request) s sine i=uCoefficient g q sine i := by
  cases sine <;> simp [Typed.uCoefficient,Typed.uToken,Typed.uLayout,uCoefficient,uToken,hq,Geometry.request]
theorem u_live (g : Geometry) (p q : Nat) (s : State α) (hp : s.pids 0=p) (hq : s.pids 1=q)
    (key : Bool) (i : ULane g) :
    Typed.ULive g.request s key i ↔ uMask g p q key i := by
  simp [Typed.ULive,Typed.uActive,Typed.uToken,Typed.uLayout,Typed.half_width,
    uMask,uCoeffMask,uToken,hp,hq,Geometry.request,and_assoc]

def uPayload (g : Geometry) (v : ChannelValues α (uInputs g)) (key odd : Bool) (i : ULane g) : α :=
  if key then if odd then v (3 : Fin 6) i else v (2 : Fin 6) i
  else if odd then v (1 : Fin 6) i else v (0 : Fin 6) i
def uCoeff (g : Geometry) (v : ChannelValues α (uInputs g)) (sine : Bool) (i : ULane g) : α :=
  if sine then v (5 : Fin 6) i else v (4 : Fin 6) i

theorem u_input (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (uInputs g)) (s : State α)
    (h : (uncachedIO rest ops g p q).InputBinding v s)
    (key odd : Bool) (i : ULane g) (live : uMask g p q key i) :
    (s.mem (region key) (uOffset g p q key odd i)).read .real=uPayload g v key odd i := by
  cases key <;> cases odd
  all_goals first
    | simpa [uncachedIO,uMemory,uInputs,mapValue,uPayload,region] using h (0 : Fin 6) i live
    | simpa [uncachedIO,uMemory,uInputs,mapValue,uPayload,region] using h (1 : Fin 6) i live
    | simpa [uncachedIO,uMemory,uInputs,mapValue,uPayload,region] using h (2 : Fin 6) i live
    | simpa [uncachedIO,uMemory,uInputs,mapValue,uPayload,region] using h (3 : Fin 6) i live

theorem u_coeff_input (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (uInputs g)) (s : State α)
    (h : (uncachedIO rest ops g p q).InputBinding v s)
    (sine : Bool) (i : ULane g) (live : uCoeffMask g q i) :
    (s.mem (coefficient sine) (uCoefficient g q sine i)).read .real=uCoeff g v sine i := by
  cases sine
  · simpa [uncachedIO,uMemory,uInputs,mapValue,uCoeff,coefficient] using h (4 : Fin 6) i live
  · simpa [uncachedIO,uMemory,uInputs,mapValue,uCoeff,coefficient] using h (5 : Fin 6) i live

theorem u_safe (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (uInputs g)) (bounds : RegionName → Nat)
    (hb : (uncachedIO rest ops g p q).InBounds bounds v) (s : State α) :
    (uInvocation rest ops g p q).Safe bounds s := by
  let t := (uInvocation rest ops g p q).initial s
  have hp : t.pids 0=p := u_pid0 rest ops g p q s
  have hq : t.pids 1=q := u_pid1 rest ops g p q s
  apply Typed.uncached_safe rest ops (Typed.uLayout g.request) t bounds
  · intro sine i ht hd
    have live : uCoeffMask g q i := by
      simpa [uCoeffMask,uToken,Typed.uToken,Typed.uLayout,Typed.half_width,hq,Geometry.request] using And.intro ht hd
    rw [u_coefficient g q t hq]
    cases sine
    · exact hb.2.1 (4 : Fin 6) i live
    · exact hb.2.1 (5 : Fin 6) i live
  · intro odd i active
    have live : uMask g p q Bool.false i := (u_live g p q t hp hq Bool.false i).mp ⟨active,by simp⟩
    rw [u_offset g p q t hp hq]
    cases odd
    · exact hb.2.1 (0 : Fin 6) i live
    · exact hb.2.1 (1 : Fin 6) i live
  · intro leader odd i active
    have live : uMask g p q Bool.true i := (u_live g p q t hp hq Bool.true i).mp ⟨active,fun _ => leader⟩
    rw [u_offset g p q t hp hq]
    cases odd
    · exact hb.2.1 (2 : Fin 6) i live
    · exact hb.2.1 (3 : Fin 6) i live

#axiomsClean u_safe
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
