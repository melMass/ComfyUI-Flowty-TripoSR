import sys
from os import path
import io

sys.path.insert(0, path.dirname(__file__))
from folder_paths import (
    get_filename_list,
    get_full_path,
    get_save_image_path,
    get_output_directory,
)
from comfy.model_management import get_torch_device
import comfy.utils
import server
from tsr.system import TSR
from PIL import Image
import numpy as np
import torch


def fill_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def restore_hook(hook):
    print("Restoring hook")
    comfy.utils.set_progress_bar_global_hook(hook)


def setup_hook(server):
    """Setup a hook for the server to support sending obj using websocket."""

    if comfy.utils.PROGRESS_BAR_HOOK is None:
        print(
            "Not sure what do to, aborting hook hijack, sending mesh using websocket won't work"
        )
        return

    print("Setting up hook")
    old_hook = comfy.utils.PROGRESS_BAR_HOOK

    def hook(value, total, preview_image):
        if preview_image is None or (preview_image and preview_image[0] not in ["OBJ"]):
            old_hook(value, total, preview_image)
            return

        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {
            "value": value,
            "max": total,
            "prompt_id": server.last_prompt_id,
            "node": server.last_node_id,
        }

        server.send_sync("progress", progress, server.client_id)

        print("Sending mesh")
        # TODO: try to send bytes directly
        server.send_sync(
            "obj_export",  # 3,  # comfy uses 1 & 2
            {"obj_export": preview_image[1]},
            server.client_id,
        )

    comfy.utils.set_progress_bar_global_hook(hook)
    return old_hook


class TripoSRModelLoader:
    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_filename_list("checkpoints"),),
                "chunk_size": ("INT", {"default": 8192, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("TRIPOSR_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Flowty TripoSR"

    def load(self, model, chunk_size):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        if not self.initialized_model:
            print("Loading TripoSR model")
            self.initialized_model = TSR.from_pretrained_custom(
                weight_path=get_full_path("checkpoints", model),
                config_path=path.join(path.dirname(__file__), "config.yaml"),
            )
            self.initialized_model.renderer.set_chunk_size(chunk_size)
            self.initialized_model.to(device)

        return (self.initialized_model,)


class TripoSRSampler:
    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRIPOSR_MODEL",),
                "reference_image": ("IMAGE",),
                "geometry_resolution": (
                    "INT",
                    {"default": 256, "min": 128, "max": 12288},
                ),
                "threshold": ("FLOAT", {"default": 25.0, "min": 0.0, "step": 0.01}),
            },
            "optional": {"reference_mask": ("MASK",)},
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "sample"
    CATEGORY = "Flowty TripoSR"

    def sample(
        self,
        model: TSR,
        reference_image: torch.Tensor,
        geometry_resolution: int,
        threshold: float,
        reference_mask=None,
    ):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        batch_size = reference_image.shape[0]
        meshes_batch = []
        for img_idx in range(batch_size):
            image = reference_image[img_idx]

            # If reference_mask is provided and has the same batch size as reference_image,
            # use the corresponding mask for the current image; otherwise, use the first mask
            if reference_mask is not None:
                if reference_mask.shape[0] == batch_size:
                    mask = reference_mask[img_idx].unsqueeze(2)
                else:
                    mask = reference_mask[0].unsqueeze(2)
                image = torch.cat((image, mask), dim=2).detach().cpu().numpy()
            else:
                image = image.detach().cpu().numpy()

            image = Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8))

            if reference_mask is not None:
                image = fill_background(image)

            image = image.convert("RGB")
            scene_codes = model([image], device)
            meshes = model.extract_mesh(
                scene_codes, resolution=geometry_resolution, threshold=threshold
            )
            meshes_batch.append(meshes[0])

        return (meshes_batch,)


class TripoSRMeshSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "prefix": ("STRING", {"default": "mesh_%batch_num%"}),
                "save_location": (
                    ["custom (server)", "comfy_output_dir", "websocket"],
                    {"default": "comfy_output_dir"},
                ),
                "save_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Flowty TripoSR"

    def save(self, mesh, prefix: str, save_location: str, save_path: str):
        saved = []
        full_output_folder = None
        filename = None

        match save_location:
            case "comfy_output_dir":
                full_output_folder, filename, counter, subfolder, filename_prefix = (
                    get_save_image_path(prefix, get_output_directory())
                )
            case "custom (server)":
                full_output_folder, filename, counter, subfolder, filename_prefix = (
                    get_save_image_path(prefix, save_path)
                )
            case "websocket":
                sinstance = server.PromptServer.instance
                if not sinstance:
                    raise ValueError("No server instance found")
                old_hook = setup_hook(sinstance)
                if not old_hook:
                    raise ValueError("No server hook found")
                try:
                    pbar = comfy.utils.ProgressBar(len(mesh))
                    step = 0

                    for batch_number, single_mesh in enumerate(mesh):
                        single_mesh.apply_transform(
                            np.array(
                                [
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1],
                                ]
                            )
                        )
                        mesh_obj = io.BytesIO()
                        single_mesh.export(mesh_obj, file_type="obj")

                        pbar.update_absolute(
                            step,
                            len(mesh),
                            (
                                "OBJ",
                                mesh_obj.getvalue().decode("utf-8"),
                                None,
                            ),
                        )
                        step += 1

                    return {}
                except Exception as e:
                    print(e)
                finally:
                    restore_hook(old_hook)
            case _:
                print("Not handled")

        if full_output_folder is None:
            raise ValueError("No output folder was found")

        full_paths = []
        for batch_number, single_mesh in enumerate(mesh):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.obj"

            single_mesh.apply_transform(
                np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            )

            mesh_path = path.join(full_output_folder, file)
            single_mesh.export(mesh_path)

            full_paths.append(mesh_path)
            saved.append({"filename": file, "type": "output", "subfolder": subfolder})

        return {"ui": {"mesh": saved}, "result": (full_paths,)}


            saved.append({"filename": file, "type": "output", "subfolder": subfolder})

        return {"ui": {"mesh": saved}}


class TripoSRViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "Flowty TripoSR"

    def display(self, mesh):
        saved = list()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path("meshsave", get_output_directory())
        )

        for batch_number, single_mesh in enumerate(mesh):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.obj"
            single_mesh.apply_transform(
                np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            )
            single_mesh.export(path.join(full_output_folder, file))
            saved.append({"filename": file, "type": "output", "subfolder": subfolder})

        return {"ui": {"mesh": saved}}


NODE_CLASS_MAPPINGS = {
    "TripoSRModelLoader": TripoSRModelLoader,
    "TripoSRSampler": TripoSRSampler,
    "TripoSRMeshSave": TripoSRMeshSave,
    "TripoSRViewer": TripoSRViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSRModelLoader": "TripoSR Model Loader",
    "TripoSRSampler": "TripoSR Sampler",
    "TripoSRMeshSave": "TripoSR Mesh Save",
    "TripoSRViewer": "TripoSR Viewer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
