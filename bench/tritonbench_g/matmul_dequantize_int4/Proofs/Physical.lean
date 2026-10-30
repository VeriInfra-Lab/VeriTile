import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Native
import bench.tritonbench_g.matmul_kernel.Proofs.Physical
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Physical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
abbrev Byte := BitVec 8
abbrev Bytes := RegionName → Nat → Byte

def encode16 (w : Word) (i : Fin 2) : Byte := (w.bits.truncate 16).extractLsb' (8*i.val) 8
def decode16 (x : Fin 2 → Byte) : Word := ⟨(x 1 ++ x 0).zeroExtend 32⟩
def decode32 (x : Fin 4 → Byte) : BitVec 32 := MatmulKernel.Physical.decode x

theorem encode_decode16 (x : Fin 2 → Byte) : encode16 (decode16 x)=x := by
  funext i
  have he : ((x 1 ++ x 0).zeroExtend 32).truncate 16=x 1 ++ x 0 := by simp
  fin_cases i
  · change (((x 1 ++ x 0).zeroExtend 32).truncate 16).extractLsb' 0 8=x 0
    rw [he]
    exact BitVec.extractLsb'_append_eq_right
  · change (((x 1 ++ x 0).zeroExtend 32).truncate 16).extractLsb' 8 8=x 1
    rw [he]
    exact BitVec.extractLsb'_append_eq_left

/-- Allocation-relative byte memory. Packed words are signed int32, whereas
A, scales and C carry exactly two little-endian bytes per element. -/
def load (m : Bytes) : State Word where
  mem := fun r a => if r="Q" ∨ r="Z" then .mk .int (decode32 (fun i => m r (4*a+i.val))).toInt
    else .mk .fp16 (decode16 (fun i => m r (2*a+i.val)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

/-- Serialize the output allocation from the executed memory. Every byte in
all other allocations is retained. The frame theorem below also retains all
holes and canaries in the output allocation. -/
def commit (initial : Bytes) (s : State Word) : Bytes := fun r b =>
  if r="C" then encode16 ((s.mem r (b/2)).read .fp16) ⟨b%2,Nat.mod_lt _ (by decide)⟩
  else initial r b

theorem commit_load (m : Bytes) : commit m (load m)=m := by
  funext r b
  by_cases hr : r="C"
  · subst r
    change encode16 (decode16 (fun i : Fin 2 => m "C" (2*(b/2)+i.val))) ⟨b%2,by omega⟩=m "C" b
    rw [show encode16 (decode16 (fun i : Fin 2 => m "C" (2*(b/2)+i.val)))=
      (fun i : Fin 2 => m "C" (2*(b/2)+i.val)) from encode_decode16 _]
    change m "C" (2*(b/2)+b%2)=m "C" b
    exact congrArg (m "C") (by omega : 2*(b/2)+b%2=b)
  · simp [commit,hr]

/-- Exact physical store corresponding to a typed half write. -/
def store16 (m : Bytes) (a : Nat) (w : Word) : Bytes := fun r b =>
  if r="C" ∧ b/2=a then encode16 w ⟨b%2,Nat.mod_lt _ (by decide)⟩ else m r b

theorem commit_write (m : Bytes) (s : State Word) (a : Nat) (w : Word) :
    commit m (s.write "C" a (.mk .fp16 w))=store16 (commit m s) a w := by
  funext r b
  by_cases hr : r="C"
  · subst r
    by_cases ha : b/2=a <;> simp [commit,store16,State.write,ha]
  · simp [commit,store16,State.write,hr]

def source (actual : Native.Instructions) (c : Call) (i : Fin 17) (m : Bytes) : Option Bytes :=
  (Native.grid actual c i (load m)).map (commit m)

def target (mma : Opcode → ISA.MMA16) (c : Call) (i : Fin 17) (m : Bytes) (x : Grid.Index (params c i)) : Word :=
  Logical.output (mma (tunings i).opcode) (params c i) (load m) x.1.val x.2.val

def Post (mma : Opcode → ISA.MMA16) (c : Call) (i : Fin 17) (initial final : Bytes) : Prop :=
  (∀ x : Grid.Index (params c i),∀ byte : Fin 2,
    final "C" (2*Geometry.address (Grid.geometry (params c i)) x+byte.val)=encode16 (target mma c i initial x) byte) ∧
  (∀ (r : RegionName) (b : Nat),(r≠"C" ∨ ∀ x : Grid.Index (params c i),b/2≠Geometry.address (Grid.geometry (params c i)) x) →
    final r b=initial r b)

theorem whole_bytes (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : Call) (hc : c.Valid) (i : Fin 17) (initial : Bytes) :
    ∃ final,source actual c i initial=some final ∧ Post mma c i initial final := by
  obtain ⟨s,he,hr,hf⟩ := Grid.whole_grid (mma (tunings i).opcode) (params c i)
    (legal c hc i) (geometry_valid c hc i)
    (by change ("A" : RegionName)≠"C"; decide) (by change ("Q" : RegionName)≠"C"; decide)
    (by change ("S" : RegionName)≠"C"; decide) (by change ("Z" : RegionName)≠"C"; decide) (load initial)
  refine ⟨commit initial s,?_,?_,?_⟩
  · simp only [source,Native.grid_transport actual mma ports,he,Option.map_some]
  · intro x byte
    have hread := hr x trivial
    have hd : (2*Geometry.address (Grid.geometry (params c i)) x+byte.val)/2=
        Geometry.address (Grid.geometry (params c i)) x := by omega
    have hm : (2*Geometry.address (Grid.geometry (params c i)) x+byte.val)%2=byte.val := by omega
    simp only [commit,ite_true,hd]
    change s.mem "C" (Geometry.address (Grid.geometry (params c i)) x)=_ at hread
    rw [hread]
    simp only [Cell.read_same,target]
    congr 1
    exact Fin.ext hm
  · intro r b outside
    by_cases hrc : r="C"
    · subst r
      have ho : ∀ x : Grid.Index (params c i),b/2≠Geometry.address (Grid.geometry (params c i)) x :=
        outside.resolve_left (by simp)
      have hmem := hf "C" (b/2) (Or.inr (fun x _ => ho x))
      have heq : commit initial s "C" b=commit initial (load initial) "C" b := by
        simp only [commit,ite_true,hmem]
      rw [heq,commit_load]
    · simp [commit,hrc]

#specNonCircular target avoiding [source,Native.grid,Grid.grid,commit]
#axiomsClean encode_decode16
#axiomsClean commit_load
#axiomsClean commit_write
#axiomsClean whole_bytes
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Physical
