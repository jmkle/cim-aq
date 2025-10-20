#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

from pathlib import Path
from typing import Any

import onnx
import torch
from brevitas.export import export_onnx_qcdq
from brevitas.nn import QuantConv2d, QuantLinear, QuantMultiheadAttention

from lib.utils.logger import logger


def has_brevitas_layers(model: torch.nn.Module) -> bool:
    """
    Check if the model contains any Brevitas quantized layers.
    
    Args:
        model (torch.nn.Module): The model to check for Brevitas layers.
    
    Returns:
        bool: True if the model contains Brevitas quantized layers, False otherwise.
    """
    brevitas_layer_types = (QuantConv2d, QuantLinear, QuantMultiheadAttention)

    for module in model.modules():
        if isinstance(module, brevitas_layer_types):
            logger.info(f"Found Brevitas layer: {type(module).__name__}")
            return True

    logger.info("No Brevitas layers found in model")
    return False


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
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

    checkpoint_dir = Path(checkpoint_path)
    artifact.add_file(str(checkpoint_dir / "model.onnx"))

    stripped_model_path = checkpoint_dir / "model_stripped.onnx"
    if stripped_model_path.exists():
        artifact.add_file(str(stripped_model_path))

    artifact.add_file(str(checkpoint_dir / "model_best.pth.tar"))
    wandb_run.log_artifact(artifact)
    logger.info('Models uploaded to wandb successfully')


def export_models(model: torch.nn.Module,
                  checkpoint_path: str,
                  inputs: torch.Tensor,
                  wandb_run: Any = None) -> None:
    """
    Export the model to ONNX format and optionally upload to Weights & Biases (wandb).
    
    Automatically detects if the model contains Brevitas quantized layers and uses the
    appropriate export method:
    - For models with Brevitas layers: uses export_onnx_qcdq and strips initializers
    - For models without Brevitas layers: uses torch.onnx.export directly

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

    output_path = Path(checkpoint_path) / Path("model.onnx")
    logger.info(f'Exporting model to {output_path}')

    # Check if model contains Brevitas layers
    use_brevitas_export = has_brevitas_layers(model)

    if use_brevitas_export:
        logger.info(
            'Using Brevitas export (export_onnx_qcdq) for quantized model')
        export_onnx_qcdq(model,
                         args=inputs,
                         export_path=output_path,
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
                         },
                         dynamo=False)
        logger.info(f'Exported quantized model to {output_path}')

        # Strip initializers from ONNX model for Brevitas exports
        strip_initializers_from_onnx(output_path)
    else:
        logger.info(
            'Using standard PyTorch ONNX export for non-quantized model')
        torch.onnx.export(model,
                          inputs,
                          output_path,
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
                          },
                          dynamo=False)
        logger.info(f'Exported standard model to {output_path}')

    if wandb_run is not None:
        upload_models_to_wandb(wandb_run, checkpoint_path)
