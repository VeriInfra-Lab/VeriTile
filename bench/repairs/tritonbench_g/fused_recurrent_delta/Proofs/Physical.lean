import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.StoredGraph
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Naturality
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.HostFrames
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Index
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Physical
open Native
noncomputable section
set_option maxHeartbeats 10000000
abbrev Cells := Memory Codec.Cell
def load (bytes : Cells) : Memory Word := fun a => Codec.decode (bytes a)
def encode (words : Memory Word) : Cells := fun a => Codec.encode (words a)
theorem materialize_hom (p : Instructions) (policy : Policy) :
    Naturality.Hom graphOps (storedOps p policy) (materialize p policy) := by
  constructor
  · rfl
  · rfl
  · intro a b;rfl
  · intro a b;rfl
  · intro a b;rfl
  · exact materialize_cut p policy

theorem materialize_input (p : Instructions) (policy : Policy) (m : Memory Word) :
    Naturality.mem (materialize p policy) (graphMemory m)=graphMemory m := rfl
/-- These calls execute every store immediately into four bytes. Subsequent
v/dk/dv loads read those cells from the current source memory. -/
def forwardGraph (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :=
  forwardCall s f (storedOps p policy) g (.bits scale) (graphMemory (load m))
def backwardGraph (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :=
  backwardCall s f (storedOps p policy) g (.bits scale) (graphMemory (load m))
def forward (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :=
  encode (evaluate p policy (forwardGraph s f g scale p policy m))
def backward (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :=
  encode (evaluate p policy (backwardGraph s f g scale p policy m))

theorem forward_graph (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :
    forwardGraph s f g scale p policy m=
      Naturality.mem (materialize p policy) (forwardCall s f graphOps g (.bits scale) (graphMemory (load m))) := by
  have he:=Naturality.forward_call (materialize_hom p policy) s f g (.bits scale) (graphMemory (load m))
  rw [materialize_input] at he
  exact he.symm
theorem backward_graph (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :
    backwardGraph s f g scale p policy m=
      Naturality.mem (materialize p policy) (backwardCall s f graphOps g (.bits scale) (graphMemory (load m))) := by
  have he:=Naturality.backward_call (materialize_hom p policy) s f g (.bits scale) (graphMemory (load m))
  rw [materialize_input] at he
  exact he.symm

theorem forward_load (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :
    load (forward s f g scale p policy m)=Native.forward s f g scale p policy (load m) := by
  funext a
  simp [forward,load,encode,forward_graph,Naturality.mem,evaluate,materialize_value,Native.forward]
theorem backward_load (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy) (m : Cells) :
    load (backward s f g scale p policy m)=Native.backward s f g scale p policy (load m) := by
  funext a
  simp [backward,load,encode,backward_graph,Naturality.mem,evaluate,materialize_value,Native.backward]

theorem forward_bytes (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy)
    (m : Cells) (a : Addr) (byte : Fin 4) :
    forward s f g scale p policy m a byte=Codec.encode (Native.forward s f g scale p policy (load m) a) byte := by
  have he:=congrFun (forward_load s f g scale p policy m) a
  have hb:=congrArg (fun x => Codec.encode x byte) he
  simpa [load] using hb
theorem backward_bytes (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy)
    (m : Cells) (a : Addr) (byte : Fin 4) :
    backward s f g scale p policy m a byte=Codec.encode (Native.backward s f g scale p policy (load m) a) byte := by
  have he:=congrFun (backward_load s f g scale p policy m) a
  have hb:=congrArg (fun x => Codec.encode x byte) he
  simpa [load] using hb

theorem forward_frame (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy)
    (m : Cells) (a : Addr) (outside : ¬MemoryProof.ForwardCallWrites s f a) :
    forward s f g scale p policy m a=m a := by
  funext byte
  rw [forward_bytes,MemoryProof.native_forward_frame s f g scale p policy (load m) a outside]
  exact congrFun (Codec.encode_decode (m a)) byte
theorem backward_frame (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions) (policy : Policy)
    (m : Cells) (a : Addr) (outside : ¬MemoryProof.BackwardWrites s f a) :
    backward s f g scale p policy m a=m a := by
  funext byte
  rw [backward_bytes,MemoryProof.native_backward_frame s f g scale p policy (load m) a outside]
  exact congrFun (Codec.encode_decode (m a)) byte

theorem forward_transport (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (hp : Obligations p) (policy : Policy) (m : Cells) :
    forward s f g scale p policy m=forward s f g scale software policy m := by
  funext a byte
  rw [forward_bytes,forward_bytes,Native.forward_transport s f g scale p hp policy]
theorem backward_transport (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (hp : Obligations p) (policy : Policy) (m : Cells) :
    backward s f g scale p policy m=backward s f g scale software policy m := by
  funext a byte
  rw [backward_bytes,backward_bytes,Native.backward_transport s f g scale p hp policy]

/-- A reload observes exactly the just-written FP32 bits, including signed
zeros and NaN payloads. No numeric assumption is used for storage. -/
theorem store_reload (p : Instructions) (policy : Policy) (e : Expr) :
    ((storedOps p policy).cut e).eval p policy=e.eval p policy := by
  simp [storedOps,Expr.eval]

#axiomsClean forward_graph
#axiomsClean backward_graph
#axiomsClean forward_load
#axiomsClean backward_load
#axiomsClean forward_bytes
#axiomsClean backward_bytes
#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean forward_transport
#axiomsClean backward_transport
#axiomsClean store_reload
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Physical
