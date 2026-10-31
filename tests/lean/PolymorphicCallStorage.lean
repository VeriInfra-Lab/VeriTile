import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PolymorphicCallStorage
open VeriTile.Triton

/-- The source has no Lean dtype directives. Its storage types come from the
call and may remain polymorphic until that call is instantiated. -/
def source (dtype out_dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",out_dtype)]} : CallSignature) in triton {
    p = tl.make_block_ptr(A, (2,), (1,), (0,), (2,), (0,))
    q = tl.make_block_ptr(B, (2,), (1,), (0,), (2,), (0,))
    x = tl.load(p, boundary_check=(0,))
    tl.store(q, x.to(q.dtype.element_ty))
  }

def explicitStorage (dtype out_dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  triton {
    tl.region A = DTYPE, B = OUT_DTYPE
    p = tl.make_block_ptr(base=A, shape=($(2)), strides=($(1)), offsets=($(0)), block_shape=($(2)), order=(0))
    q = tl.make_block_ptr(base=B, shape=($(2)), strides=($(1)), offsets=($(0)), block_shape=($(2)), order=(0))
    x = tl.load(p, boundary_check=([0] : List Nat))
    tl.store(q, ((x).to(OUT_DTYPE)).to(OUT_DTYPE))
  }

theorem complete_program (dtype out_dtype : FloatDType) (A B : RegionName) :
    source dtype out_dtype A B=explicitStorage dtype out_dtype A B := by rfl

theorem memory_ports (dtype out_dtype : FloatDType) (A B : RegionName) :
    (source dtype out_dtype A B).inputs=[A] ∧
      (source dtype out_dtype A B).outputs=[B] := by exact ⟨rfl,rfl⟩

/-- Equal storage expressions share one type identity; this makes ordinary
register assignments and same-dtype arithmetic well-typed. -/
def sameStorage (dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",dtype)]} : CallSignature) in triton {
    x = tl.load(A)
    y = tl.load(B)
    z = x + y
    tl.store(B, z)
  }

/-- A cast immediately after a load still reads the original region. -/
def castLoad (dtype out_dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",out_dtype)]} : CallSignature) in triton {
    p = tl.make_block_ptr(A, (2,), (1,), (0,), (2,), (0,))
    q = tl.make_block_ptr(B, (2,), (1,), (0,), (2,), (0,))
    x = tl.load(p, boundary_check=(0,)).to(tl.float32)
    tl.store(q, x.to(q.dtype.element_ty))
  }

theorem cast_load_ports (dtype out_dtype : FloatDType) (A B : RegionName) :
    (castLoad dtype out_dtype A B).inputs=[A] ∧
      (castLoad dtype out_dtype A B).outputs=[B] := by exact ⟨rfl,rfl⟩

/-- A source register dtype query retains its explicit conversion node. -/
def registerCast (dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",dtype)]} : CallSignature) in triton {
    x = tl.load(A)
    y = x.to(x.dtype)
    tl.store(B, y)
  }

theorem register_cast_node (dtype : FloatDType) (A B : RegionName) :
    ((registerCast dtype A B).surfaceBody[1]?) =
      some (.assign dtype.toTileDType [] "y"
        (.alg (.castFloat dtype dtype
          (.withPrecision dtype .fp32 (.ref dtype.toTileDType [] "x"))))) := by rfl

/-- Dtype tests select a compile-time branch while the pasted source keeps
both casts and the common store, as in the repaired matmul. -/
def dtypeBranch (dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",dtype)]} : CallSignature) in triton {
    x = tl.load(A)
    if (B.dtype.element_ty == tl.float8e4nv):
        y = x.to(tl.float8e4nv)
    else:
        y = x.to(tl.float16)
    tl.store(B, y)
  }

def fp8Branch (dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",dtype)]} : CallSignature) in triton {
    x = tl.load(A)
    y = x.to(tl.float8e4nv)
    tl.store(B, y)
  }

def fp16Branch (dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({floatingPointers := [("A",dtype),("B",dtype)]} : CallSignature) in triton {
    x = tl.load(A)
    y = x.to(tl.float16)
    tl.store(B, y)
  }

theorem dtype_branch_exact (dtype : FloatDType) (A B : RegionName) :
    dtypeBranch dtype A B=
      if dtype=.f8e4 then fp8Branch dtype A B else fp16Branch dtype A B := by rfl

/-- error: pointer parameter B is missing from the call signature -/
#guard_msgs in
def rejectMissingStorage (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.fp32)]} : CallSignature) in triton {
    q = tl.make_block_ptr(B, (2,), (1,), (0,), (2,), (0,))
    x = tl.load(A)
    tl.store(q, x)
  }

#axiomsClean complete_program
#axiomsClean memory_ports
#axiomsClean cast_load_ports
#axiomsClean register_cast_node
#axiomsClean dtype_branch_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.PolymorphicCallStorage
