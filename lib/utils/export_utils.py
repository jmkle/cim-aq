from pathlib import Path
from typing import Any

import onnx
import torch
from brevitas.export import export_onnx_qcdq

from lib.utils.logger import logger


def load_best_model(model: torch.nn.Module,
                    checkpoint_path: str) -> torch.nn.Module:
    """
    Load the best model from the checkpoint.
    
    Args:
        model (torch.nn.Module): The model instance to load the state dict into.
        checkpoint_path (str): Path to the directory containing the checkpoint file.
                              Expected to contain 'model_best.pth.tar'.
    
    Returns:
        torch.nn.Module: The model with loaded state dict from the checkpoint.
    
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        KeyError: If the checkpoint doesn't contain 'state_dict' key.
        RuntimeError: If there's a mismatch between model architecture and checkpoint.
    """
    logger.info(f'Loading best model from {checkpoint_path}')
    model_path = Path(checkpoint_path) / Path("model_best.pth.tar")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def strip_initializers_from_onnx(onnx_model_path: Path) -> None:
    """
    Strip initializers from ONNX model.
    This function removes initializers from the inputs of an ONNX model
    if the model's IR version is 4 or higher. It saves the modified model
    with a '_stripped' suffix in the same directory.

    Args:
        onnx_model_path (Path): Path to the ONNX model file.

    Returns:
        None: The function modifies the ONNX model in place and saves it with '_stripped' suffix.
    
    Raises:
        Exception: If there is an error during the ONNX model loading or processing.
    """
    try:
        logger.info(f"Strip initializers from ONNX model")

        model = onnx.load(onnx_model_path)

        if model.ir_version < 4:
            logger.error(
                "Model with ir_version below 4 requires initializers in graph inputs"
            )
            return

        # Remove initializers from inputs
        inputs = model.graph.input
        name_to_input = {inp.name: inp for inp in inputs}
        for init in model.graph.initializer:
            if init.name in name_to_input:
                inputs.remove(name_to_input[init.name])

        stripped_path = onnx_model_path.parent / "model_stripped.onnx"
        onnx.save(model, stripped_path)
        logger.info(f"Stripped ONNX initializers → {stripped_path}")
    except Exception as e:
        logger.error(f"Error processing initializers: {e}")


def upload_models_to_wandb(wandb_run: Any, checkpoint_path: str) -> None:
    """
    Upload the exported ONNX models and the best checkpoint to Weights & Biases (wandb).

    Args:
        wandb_run (Any): The wandb run instance to log the artifacts.
        checkpoint_path (str): Path to the directory containing the exported ONNX model.
    
    Returns:
        None
    """
    from wandb import Artifact, init

    logger.info(f'Uploading models to wandb from {checkpoint_path}')
    artifact = Artifact('model', type='model')
    artifact.add_file(str(Path(checkpoint_path) / Path("model.onnx")))
    artifact.add_file(str(Path(checkpoint_path) / Path("model_stripped.onnx")))
    artifact.add_file(str(Path(checkpoint_path) / Path("model_best.pth.tar")))
    wandb_run.log_artifact(artifact)
    logger.info('Models uploaded to wandb successfully')


def export_models(model: torch.nn.Module,
                  checkpoint_path: str,
                  inputs: torch.Tensor,
                  wandb_run: Any = None) -> None:
    """
    Export the model to ONNX format and optionally upload to Weights & Biases (wandb).

    Args:
        model (torch.nn.Module): The model to be exported.
        checkpoint_path (str): Path to the directory where the model will be saved.
        inputs (torch.Tensor): Sample input tensor for the model.
        wandb_run (Any, optional): The wandb run instance for logging artifacts. Defaults to None.

    Returns:
        None: The function saves the ONNX model to the specified path and optionally uploads it to wandb.

    Raises:
        Exception: If there is an error during the ONNX export or wandb upload.
    """
    model = load_best_model(model, checkpoint_path)
    model = model.cpu()
    inputs = inputs.cpu()
    # Ensure model is in eval and export mode for Brevitas
    model.eval()
    with torch.no_grad():
        _ = model(inputs)
    logger.info(f'Exporting model to {checkpoint_path}/model.onnx')
    export_onnx_qcdq(model,
                     args=inputs,
                     export_path=Path(checkpoint_path) / Path("model.onnx"),
                     opset_version=20,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={
                         'input': {
                             0: 'batch_size'
                         },
                         'output': {
                             0: 'batch_size'
                         }
                     })
    logger.info(f'Exported model to {checkpoint_path}/model.onnx')

    strip_initializers_from_onnx(Path(checkpoint_path) / Path("model.onnx"))

    if wandb_run is not None:
        upload_models_to_wandb(wandb_run, checkpoint_path)
