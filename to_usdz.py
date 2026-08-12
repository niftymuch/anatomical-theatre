"""GLB -> USDZ for AR Quick Look.  Y-up, metres, UsdPreviewSurface materials."""
import os, sys, shutil, tempfile
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils


def convert(glb_path, usdz_path):
    scene = trimesh.load(glb_path)
    work = tempfile.mkdtemp()
    usd_path = os.path.join(work, "model.usdc")

    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Theatre")
    stage.SetDefaultPrim(root.GetPrim())
    looks = UsdGeom.Scope.Define(stage, "/Theatre/Looks")

    # any texture the glTF references has to travel inside the package
    here = os.path.dirname(os.path.abspath(glb_path))
    tex_src = os.path.join(here, "brick_diffuse.png")
    tex_name = "brick_diffuse.png"
    has_tex = os.path.exists(tex_src)
    if has_tex:
        shutil.copy(tex_src, os.path.join(work, tex_name))

    made = {}
    for name, g in scene.geometry.items():
        safe = "".join(c if c.isalnum() else "_" for c in name)
        mat_name = safe.split("_")[0]

        # ---- material, once per palette entry ----
        if mat_name not in made:
            mpath = f"/Theatre/Looks/{mat_name}"
            mat = UsdShade.Material.Define(stage, mpath)
            sh = UsdShade.Shader.Define(stage, mpath + "/Surface")
            sh.CreateIdAttr("UsdPreviewSurface")
            c = np.array(g.visual.material.baseColorFactor[:3]) / 255.0
            a = float(g.visual.material.baseColorFactor[3]) / 255.0
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                float(g.visual.material.roughnessFactor or 0.85))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if a < 0.99:
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(a)

            uv0 = getattr(g.visual, "uv", None)
            if has_tex and uv0 is not None and len(uv0):
                reader = UsdShade.Shader.Define(stage, mpath + "/stReader")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                tex = UsdShade.Shader.Define(stage, mpath + "/diffuseTex")
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{tex_name}")
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    reader.ConnectableAPI(), "result")
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")

            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            made[mat_name] = mat

        # ---- mesh ----
        mesh = UsdGeom.Mesh.Define(stage, f"/Theatre/{safe}")
        v = np.asarray(g.vertices, dtype=np.float32)
        f = np.asarray(g.faces, dtype=np.int32)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v.tolist()])
        mesh.CreateFaceVertexCountsAttr([3] * len(f))
        mesh.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        n = np.asarray(g.vertex_normals, dtype=np.float32)
        mesh.CreateNormalsAttr([Gf.Vec3f(*p) for p in n.tolist()])
        mesh.SetNormalsInterpolation("vertex")
        ext = np.array([v.min(axis=0), v.max(axis=0)], dtype=np.float32)
        mesh.CreateExtentAttr([Gf.Vec3f(*ext.tolist()[0]), Gf.Vec3f(*ext.tolist()[1])])
        uv = getattr(g.visual, "uv", None)
        if uv is not None and len(uv) == len(v):
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            # glTF puts the origin top-left, USD bottom-left
            st.Set([Gf.Vec2f(float(a), float(1.0 - b)) for a, b in uv.tolist()])
        UsdShade.MaterialBindingAPI(mesh).Bind(made[mat_name])

    stage.GetRootLayer().Save()
    if os.path.exists(usdz_path):
        os.remove(usdz_path)
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(usd_path), usdz_path)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{usdz_path}  {os.path.getsize(usdz_path)/1024:.0f} KB")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    convert(os.path.join(here, "theatre.glb"), os.path.join(here, "theatre.usdz"))
    convert(os.path.join(here, "theatre_model.glb"), os.path.join(here, "theatre_model.usdz"))
