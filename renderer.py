# renderer.py
from __future__ import annotations
import argparse
import numpy as np
import sapien
from pathlib import Path

def create_engine_and_scene(bg_color=(1.0, 1.0, 1.0, 1.0), offscreen=False):
    engine = sapien.Engine()

    # Prefer SapienRenderer; fall back to whatever is available
    RendererClass = None
    # SAPIEN 3.x
    if hasattr(sapien, "SapienRenderer"):
        RendererClass = sapien.SapienRenderer
    # SAPIEN 2.x 仍然可以从 core 里拿 VulkanRenderer，但这里统一用 SapienRenderer 包装器
    if RendererClass is None and hasattr(sapien, "VulkanRenderer"):
        # 新版本会给出 rename 警告，但仍可用
        RendererClass = sapien.VulkanRenderer

    renderer = RendererClass(offscreen_only=offscreen) if RendererClass else sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1/240.0)

    # 背景色设置：不同版本 API 不同，做安全判断
    # 1) viewer 或 render scene 上可能有 set_background_color / clear color
    # 2) 某些版本只能通过 viewer.window 来设置
    def set_background_color(viewer_or_scene, rgba):
        for attr in ["set_background_color", "set_clear_color", "set_clear_color_rgba"]:
            if hasattr(viewer_or_scene, attr):
                try:
                    getattr(viewer_or_scene, attr)(rgba)
                    return
                except Exception:
                    pass

    return engine, renderer, scene, set_background_color, bg_color

def load_urdf(scene, urdf_path: str):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # 按需把 URDF 加载成可动或刚体；这里一般是关节对象
    art = loader.load(urdf_path)
    if art is None:
        # 退而求其次，尝试以“kinematic”方式载入
        art = loader.load_kinematic(urdf_path)
    if art is None:
        raise RuntimeError(f"Failed to load URDF: {urdf_path}")
    return art

def force_diffuse_only(scene, keep_texture: bool):
    """
    遍历所有渲染形状，把材质强制设为“纯漫反射”：
      metallic=0, roughness=1, specular=0，
      去掉会导致高光/反射的贴图；仅可选保留 base color。
    """
    rscene = scene.get_renderer_scene()

    # 环境光偏亮一点，避免你关掉高光后场景过暗
    if hasattr(rscene, "set_ambient_light"):
        rscene.set_ambient_light([0.7, 0.7, 0.7])

    # 你也可以不加任何直射光，只靠环境光；如果想加一盏很柔的方向光：
    if hasattr(rscene, "add_directional_light"):
        rscene.add_directional_light(direction=[0, -1, -1], color=[0.3, 0.3, 0.3], shadow=False)

    # 遍历所有实体，拿到渲染组件后改材质
    # SAPIEN 3.x 通常：entity.get_components(sapien.render.RenderShapeComponent)
    # 为保证兼容，这里尽量通过 wrapper 访问：
    try:
        import sapien.render as srender  # pysapien.render
        RenderShapeComponent = getattr(srender, "RenderShapeComponent", None)
        RenderMaterial = getattr(srender, "RenderMaterial", None)
    except Exception:
        RenderShapeComponent = None
        RenderMaterial = None

    entities = scene.get_all_entities() if hasattr(scene, "get_all_entities") else []

    for ent in entities:
        comps = []
        if RenderShapeComponent is not None and hasattr(ent, "get_components"):
            try:
                comps = ent.get_components(RenderShapeComponent)
            except Exception:
                comps = []
        # 某些版本还有 get_render_shapes()
        if not comps and hasattr(ent, "get_render_shapes"):
            try:
                comps = ent.get_render_shapes()
            except Exception:
                comps = []

        for comp in comps:
            # 不同版本：comp.material 或 comp.get_material()
            mat = None
            if hasattr(comp, "material"):
                mat = comp.material
            elif hasattr(comp, "get_material"):
                mat = comp.get_material()

            if mat is None:
                continue

            # 只保留 base color（如果 keep_texture），禁用其它会引入高光/反射的贴图
            if keep_texture:
                # 仅保留 base color texture；其他全部移除
                for tex_attr in [
                    "normal_texture", "metallic_texture", "roughness_texture",
                    "specular_texture", "emission_texture", "clearcoat_texture",
                ]:
                    if hasattr(mat, tex_attr):
                        setattr(mat, tex_attr, None)
            else:
                # 不保留任何纹理
                for tex_attr in [
                    "base_color_texture", "diffuse_texture",  # 兼容旧名
                    "normal_texture", "metallic_texture", "roughness_texture",
                    "specular_texture", "emission_texture", "clearcoat_texture",
                ]:
                    if hasattr(mat, tex_attr):
                        setattr(mat, tex_attr, None)

            # 颜色（基础反照率）保持或设为一个中性灰
            if hasattr(mat, "base_color"):
                if not keep_texture:
                    mat.base_color = [0.8, 0.8, 0.8, 1.0]
            elif hasattr(mat, "diffuse_color"):
                if not keep_texture:
                    mat.diffuse_color = [0.8, 0.8, 0.8, 1.0]

            # ——关键：彻底去镜面——
            for attr, val in [
                ("metallic", 0.0),
                ("roughness", 1.0),
                ("specular", 0.0),
                ("clearcoat", 0.0),
                ("ior", 1.0),  # 折射率（有些 shader 会用），尽量降低反射
            ]:
                if hasattr(mat, attr):
                    setattr(mat, attr, val)

            # 额外保险：有的 shader 会把“光泽度/Glossiness=1-roughness”
            if hasattr(mat, "glossiness"):
                mat.glossiness = 0.0  # = 1 - roughness(=1)

def center_camera_on_articulation(viewer, art):
    # 粗略把相机对准 URDF
    aabb = art.compute_global_aabb() if hasattr(art, "compute_global_aabb") else None
    if aabb is not None:
        lower = np.array(aabb[0]); upper = np.array(aabb[1])
        center = (lower + upper) * 0.5
        extent = np.linalg.norm(upper - lower)
    else:
        center = np.array([0, 0, 0]); extent = 2.0

    # 尝试用 viewer 的便捷接口
    if hasattr(viewer, "set_camera_xyz") and hasattr(viewer, "set_camera_rpy"):
        viewer.set_camera_xyz(center[0] + 2.0*max(1.0, extent), center[1] + 0.5*extent, center[2] + 0.5*extent)
        viewer.set_camera_rpy(0, -0.2, np.pi)  # 俯仰一点
    elif hasattr(viewer, "set_camera_pose"):
        from transforms3d.euler import euler2quat
        q = euler2quat(0, -0.2, np.pi)
        viewer.set_camera_pose(sapien.Pose([center[0] + 2.0*max(1.0, extent), center[1], center[2]], q))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=str, required=True)
    ap.add_argument("--background", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    ap.add_argument("--keep-texture", action="store_true", help="仅保留 base color 纹理，禁用其余反射相关纹理")
    ap.add_argument("--offscreen", action="store_true", help="离屏渲染（不弹 viewer 窗口）")
    args = ap.parse_args()

    engine, renderer, scene, set_bg, bg_rgba = create_engine_and_scene(tuple(args.background)+(1.0,), args.offscreen)

    # 加载 URDF
    art = load_urdf(scene, args.urdf)

    # 施加“纯漫反射”策略
    force_diffuse_only(scene, keep_texture=args.keep_texture)

    # 创建 viewer，设置背景色（尽力而为的兼容做法）
    viewer = scene.create_viewer() if hasattr(scene, "create_viewer") else None
    if viewer is not None:
        # 有的版本暴露 viewer.render_scene / viewer.window
        if hasattr(viewer, "render_scene") and viewer.render_scene is not None:
            set_bg(viewer.render_scene, bg_rgba)
        if hasattr(viewer, "window"):
            set_bg(viewer.window, bg_rgba)
        center_camera_on_articulation(viewer, art)
        viewer.set_scene(scene) if hasattr(viewer, "set_scene") else None
        viewer.loop()
    else:
        # 离屏拍一张（示例：如果你需要）
        # 在 2.x/3.x 下的相机 API 差别较大，这里仅示范：
        cam = scene.add_mounted_camera(
            name="main", mount=scene, pose=sapien.Pose([2, 0, 1], [1,0,0,0]),
            width=1280, height=720, fovy=35, near=0.1, far=100
        ) if hasattr(scene, "add_mounted_camera") else None
        if cam:
            scene.update_render()
            cam.take_picture()
            import imageio
            rgba = cam.get_color_rgba()  # h,w,4 float32
            imageio.imwrite("output.png", (np.clip(rgba, 0, 1)*255).astype(np.uint8))
            print("[info] saved output.png")

if __name__ == "__main__":
    main()
