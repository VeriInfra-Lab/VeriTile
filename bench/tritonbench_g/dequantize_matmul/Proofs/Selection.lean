import bench.tritonbench_g.dequantize_matmul.Proofs.Host

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 3000000

inductive LibraryFamily where
  | scalarFP32
  | mmaSyncM16N8K8
  | wgmmaM64N8K8
  deriving DecidableEq

def LibraryFamily.instruction : LibraryFamily → Option MMAKind
  | .scalarFP32 => none
  | .mmaSyncM16N8K8 => some .mmaSyncM16N8K8RowColF32TF32TF32F32
  | .wgmmaM64N8K8 => some .wgmmaM64N8K8F32TF32TF32

def Plan.supported (family : LibraryFamily) : Plan K → Prop
  | .zero | .product _ => True
  | .fma _ seed => seed.supported family
  | .add left right => left.supported family ∧ right.supported family
  | .mma8 kind _ _ seed => family.instruction=some kind ∧ seed.supported family

/-- This metadata has no input/output numeric values. The installed library
must select its implementation/plan from these ABI/configuration fields. -/
structure LibraryMetadata where
  libraryBuild : String
  algorithmId : Nat
  device : Nat
  architecture : Nat
  aRowStride : Nat
  aColumnStride : Nat
  bRowStride : Nat
  bColumnStride : Nat
  outRowStride : Nat
  outColumnStride : Nat
  workspaceBytes : Nat
  family : LibraryFamily

/-- Trusted library selection and local primitive lowering are ordinary
implementation boundaries. This interface does not verify cuBLAS internals.
It cannot select a plan by receiving a completed output or input matrix. -/
structure LibraryImplementation where
  select : (s : Shape) → LibraryMetadata → LibraryPlan s
  supported : ∀ s metadata r,s.M>r → ∀ n,s.N>n →
    ((select s metadata).output r n).supported metadata.family

structure LaunchConfig where
  blockK : Nat
  blockN : Nat
  warps : Nat
  stages : Nat
  deriving DecidableEq

def launchConfigs : List LaunchConfig :=
  [⟨128,128,4,3⟩,⟨256,64,8,3⟩,⟨256,32,4,4⟩,⟨128,32,4,4⟩,
   ⟨64,32,4,4⟩,⟨128,32,4,4⟩,⟨32,32,4,4⟩,⟨32,32,2,5⟩,
   ⟨64,32,2,5⟩,⟨64,256,8,3⟩,⟨32,256,4,4⟩,⟨32,128,4,4⟩,
   ⟨32,64,4,4⟩,⟨32,128,4,4⟩,⟨32,32,4,4⟩,⟨32,32,2,5⟩,⟨32,64,2,5⟩]

theorem launchConfigs_positive : ∀ c∈launchConfigs,0<c.blockK ∧ 0<c.blockN := by
  decide

def LaunchConfig.tile (c : LaunchConfig) (member : c∈launchConfigs) : Tile :=
  ⟨c.blockK,c.blockN,(launchConfigs_positive c member).1,(launchConfigs_positive c member).2⟩

def Plan.scalar (K : Nat) : Plan K :=
  (List.finRange K).foldr Plan.fma .zero

theorem foldr_terms (xs : List (Fin K)) :
    (xs.foldr Plan.fma .zero).terms=xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [Plan.terms,ih]

theorem Plan.scalar_complete (K : Nat) : (Plan.scalar K).Complete := by
  unfold Plan.scalar Plan.Complete
  rw [foldr_terms]

theorem foldr_supported (xs : List (Fin K)) (family : LibraryFamily) :
    (xs.foldr Plan.fma .zero).supported family := by
  induction xs with
  | nil => trivial
  | cons x xs ih => exact ih

/-- A witness that the structural requirements are satisfiable. This is not
an assertion that torch.mm selected this scalar contraction on any device. -/
def scalarWitness : LibraryImplementation where
  select := fun s _ => ⟨fun _ _ => Plan.scalar s.K,fun _ _ _ _ => Plan.scalar_complete s.K⟩
  supported := fun _ _ _ _ _ _ => foldr_supported _ _

#axiomsClean launchConfigs_positive
#axiomsClean Plan.scalar_complete
#axiomsClean scalarWitness
end
end VeriTile.Bench.Repairs.DequantizeMatmul
