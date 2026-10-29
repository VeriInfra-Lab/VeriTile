import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem c_input_offset (g : Geometry) (p q : Nat) (s : State α) (hp : s.pids 0=p) (hq : s.pids 1=q)
    (key odd : Bool) (i : CLane g) :
    Typed.cInput (Typed.cLayout g.request) s key odd i=cInput g p q key odd i := by
  cases key <;> simp [Typed.cInput,Typed.cDim,Typed.cLayout,cInput,Request.heads,
    Typed.half_width,hp,hq,Geometry.request]
theorem c_coefficient (g : Geometry) (q : Nat) (s : State α) (hq : s.pids 1=q)
    (sine : Bool) (i : CLane g) :
    Typed.cCoefficient (Typed.cLayout g.request) s sine i=cCoefficient g q sine i := by
  cases sine <;> simp [Typed.cCoefficient,Typed.cLayout,cCoefficient,hq,Geometry.request]
theorem c_active (g : Geometry) (p : Nat) (s : State α) (hp : s.pids 0=p)
    (key : Bool) (i : CLane g) :
    (p<g.qHeads ∧ Typed.cActive (Typed.cLayout g.request) s key i) ↔ cMask g p key i := by
  simp [Typed.cActive,Typed.cLayout,Typed.half_width,cMask,cCoeffMask,hp,Geometry.request]
  tauto

def cPayload (g : Geometry) (v : ChannelValues α (cInputs g)) (key odd : Bool) (i : CLane g) : α :=
  if key then if odd then v (3 : Fin 8) i else v (2 : Fin 8) i
  else if odd then v (1 : Fin 8) i else v (0 : Fin 8) i
def cCoeff (g : Geometry) (v : ChannelValues α (cInputs g)) (sine : Bool) (i : CLane g) : α :=
  if sine then v (5 : Fin 8) i else v (4 : Fin 8) i

theorem c_input (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (cInputs g)) (s : State α)
    (h : (cachedIO rest ops g p q).InputBinding v s)
    (key odd : Bool) (i : CLane g) (live : cMask g p key i) :
    (s.mem (region key) (cInput g p q key odd i)).read .real=cPayload g v key odd i := by
  cases key <;> cases odd
  all_goals first
    | simpa [cachedIO,cMemory,cInputs,mapValue,cPayload,region] using h (0 : Fin 8) i live
    | simpa [cachedIO,cMemory,cInputs,mapValue,cPayload,region] using h (1 : Fin 8) i live
    | simpa [cachedIO,cMemory,cInputs,mapValue,cPayload,region] using h (2 : Fin 8) i live
    | simpa [cachedIO,cMemory,cInputs,mapValue,cPayload,region] using h (3 : Fin 8) i live

theorem c_coeff_input (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (cInputs g)) (s : State α)
    (h : (cachedIO rest ops g p q).InputBinding v s)
    (sine : Bool) (i : CLane g) (live : cCoeffMask g p i) :
    (s.mem (coefficient sine) (cCoefficient g q sine i)).read .real=cCoeff g v sine i := by
  cases sine
  · simpa [cachedIO,cMemory,cInputs,mapValue,cCoeff,coefficient] using h (4 : Fin 8) i live
  · simpa [cachedIO,cMemory,cInputs,mapValue,cCoeff,coefficient] using h (5 : Fin 8) i live

theorem c_metadata (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (v : ChannelValues α (cInputs g)) (s : State α)
    (h : (cachedIO rest ops g p q).InputBinding v s) (head : p<g.qHeads) :
    Typed.cPast (Typed.cLayout g.request) ((cInvocation rest ops g p q).initial s)=past g v ∧
    Typed.cBlock (Typed.cLayout g.request) ((cInvocation rest ops g p q).initial s)=page g v := by
  have hl : (s.mem "Lengths" q).read .int=contextLength g v := h (6 : Fin 8) PUnit.unit head
  have ht : (s.mem "Tables" (tableOffset g q v)).read .int=page g v := h (7 : Fin 8) PUnit.unit head
  have hpast : Typed.cPast (Typed.cLayout g.request) ((cInvocation rest ops g p q).initial s)=past g v := by
    simp [Typed.cPast,Typed.cLayout,past,Geometry.request,hl]
  refine ⟨hpast,?_⟩
  unfold Typed.cBlock
  rw [hpast]
  simpa [Typed.cLayout,tableOffset,Geometry.request] using ht

theorem c_output_offset (g : Geometry) (p q : Nat) (s : State α) (hp : s.pids 0=p) (hq : s.pids 1=q)
    (v : ChannelValues α (cInputs g))
    (hpast : Typed.cPast (Typed.cLayout g.request) s=past g v)
    (hpage : Typed.cBlock (Typed.cLayout g.request) s=page g v)
    (key odd : Bool) (i : CLane g) :
    Typed.cOffset (Typed.cLayout g.request) s key odd i=
      if key then cacheOffset g p v odd i else cInput g p q Bool.false odd i := by
  cases key
  · exact c_input_offset g p q s hp hq Bool.false odd i
  · simp only [Typed.cOffset,↓reduceIte,hpast,hpage]
    simp [Typed.cLayout,Typed.cDim,Typed.half_width,cacheOffset,hp,Geometry.request]

theorem c_safe (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (token : q<g.tokens) (v : ChannelValues α (cInputs g)) (bounds : RegionName → Nat)
    (hb : (cachedIO rest ops g p q).InBounds bounds v) (s : State α)
    (h : (cachedIO rest ops g p q).InputBinding v s) :
    (cInvocation rest ops g p q).Safe bounds s := by
  let t := (cInvocation rest ops g p q).initial s
  have hp : t.pids 0=p := c_pid0 rest ops g p q s
  have hq : t.pids 1=q := c_pid1 rest ops g p q s
  apply Typed.cached_safe rest ops (Typed.cLayout g.request) t bounds
  · simpa [hq,Typed.cLayout,Geometry.request] using token
  · intro head
    have head' : p<g.qHeads := by simpa [hp,Typed.cLayout,Geometry.request] using head
    simpa [hq] using hb.2.1 (6 : Fin 8) PUnit.unit head'
  · intro head
    have head' : p<g.qHeads := by simpa [hp,Typed.cLayout,Geometry.request] using head
    have md := c_metadata rest ops g p q v s h head'
    rw [md.1]
    simpa [Typed.cLayout,Geometry.request,hq,tableOffset] using hb.2.1 (7 : Fin 8) PUnit.unit head'
  · intro head sine i hd
    have live : cCoeffMask g p i := by
      simpa [cCoeffMask,Typed.cLayout,Typed.half_width,hp,Geometry.request] using And.intro head hd
    rw [c_coefficient g q t hq]
    cases sine
    · exact hb.2.1 (4 : Fin 8) i live
    · exact hb.2.1 (5 : Fin 8) i live
  · intro head key odd i active
    have head' : p<g.qHeads := by simpa [hp,Typed.cLayout,Geometry.request] using head
    have live : cMask g p key i := (c_active g p t hp key i).mp ⟨head',active⟩
    rw [c_input_offset g p q t hp hq]
    cases key <;> cases odd
    all_goals first
      | exact hb.2.1 (0 : Fin 8) i live
      | exact hb.2.1 (1 : Fin 8) i live
      | exact hb.2.1 (2 : Fin 8) i live
      | exact hb.2.1 (3 : Fin 8) i live
  · intro head key odd i active
    have head' : p<g.qHeads := by simpa [hp,Typed.cLayout,Geometry.request] using head
    have live : cMask g p key i := (c_active g p t hp key i).mp ⟨head',active⟩
    have md := c_metadata rest ops g p q v s h head'
    rw [c_output_offset g p q t hp hq v md.1 md.2]
    cases key <;> cases odd
    all_goals first
      | exact hb.2.2.1 (0 : Fin 4) i live
      | exact hb.2.2.1 (1 : Fin 4) i live
      | exact hb.2.2.1 (2 : Fin 4) i live
      | exact hb.2.2.1 (3 : Fin 4) i live

#axiomsClean c_safe
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
