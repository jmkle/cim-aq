#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

import argparse
import logging
from pathlib import Path

import torch
import torchvision.models as tv_models
from torch import nn

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torchvision_pretrained_model(model_name, num_classes):
    """
    Get a pre-trained model from torchvision based on model name.
    Returns the model or None if the model isn't available.
    
    Properly initializes the classifier layer when the number of classes
    is different from ImageNet's 1000 classes to prevent training issues.
    """

    try:
        if model_name is None:
            logger.warning(f"No torchvision mapping found for {model_name}")
            return None

        logger.info(f"Loading pretrained {model_name} from torchvision...")

        # Get the model with pretrained weights
        model_fn = getattr(tv_models, model_name, None)
        if model_fn is None:
            logger.warning(
                f"No model named {model_name} in torchvision.models")
            return None

        pretrained_model = model_fn(weights="DEFAULT")

        # If num_classes is different from ImageNet's 1000 classes, adjust the model
        if num_classes != 1000:
            logger.info(
                f"Adjusting pretrained model for {num_classes} classes (was: 1000)"
            )

            # Handle inception models
            if model_name.startswith('inception') and hasattr(
                    pretrained_model, 'fc'):
                in_features = pretrained_model.fc.in_features
                pretrained_model.fc = nn.Linear(in_features, num_classes)
                nn.init.normal_(pretrained_model.fc.weight, 0, 0.01)
                nn.init.constant_(pretrained_model.fc.bias, 0)
                logger.info(
                    f"Inception fc layer initialized: {in_features} -> {num_classes} with normal(0, 0.01)"
                )

            # ResNet, DenseNet, etc. with fc layer
            elif hasattr(pretrained_model, 'fc'):
                in_features = pretrained_model.fc.in_features
                pretrained_model.fc = nn.Linear(in_features, num_classes)
                # Proper initialization
                nn.init.normal_(pretrained_model.fc.weight, 0, 0.01)
                nn.init.constant_(pretrained_model.fc.bias, 0)
                logger.info(
                    f"FC layer initialized: {in_features} -> {num_classes} with normal(0, 0.01)"
                )

            # MobileNet, EfficientNet etc. with classifier as single Linear layer
            elif hasattr(pretrained_model, 'classifier') and isinstance(
                    pretrained_model.classifier, nn.Linear):
                in_features = pretrained_model.classifier.in_features
                pretrained_model.classifier = nn.Linear(
                    in_features, num_classes)
                nn.init.normal_(pretrained_model.classifier.weight, 0, 0.01)
                nn.init.constant_(pretrained_model.classifier.bias, 0)
                logger.info(
                    f"Classifier initialized: {in_features} -> {num_classes} with normal(0, 0.01)"
                )

            # Models with sequential classifier (VGG, SqueezeNet, etc.)
            elif hasattr(pretrained_model, 'classifier') and isinstance(
                    pretrained_model.classifier, nn.Sequential):
                # For squeezenet the classifier includes a conv layer
                if 'squeezenet' in model_name:
                    pretrained_model.classifier[1] = nn.Conv2d(
                        pretrained_model.classifier[1].in_channels,
                        num_classes,
                        kernel_size=1)
                    # Initialize final conv layer
                    nn.init.normal_(pretrained_model.classifier[1].weight, 0,
                                    0.01)
                    nn.init.constant_(pretrained_model.classifier[1].bias, 0)
                    logger.info(
                        f"SqueezeNet classifier initialized for {num_classes} classes"
                    )
                # For VGG and other models with sequential classifier ending with Linear layer
                else:
                    last_layer = pretrained_model.classifier[-1]
                    if isinstance(last_layer, nn.Linear):
                        in_features = last_layer.in_features
                        pretrained_model.classifier[-1] = nn.Linear(
                            in_features, num_classes)
                        nn.init.normal_(pretrained_model.classifier[-1].weight,
                                        0, 0.01)
                        nn.init.constant_(pretrained_model.classifier[-1].bias,
                                          0)
                        model_type = "VGG" if model_name.startswith(
                            'vgg') else "Sequential"
                        logger.info(
                            f"{model_type} classifier's last layer initialized: {in_features} -> {num_classes} with normal(0, 0.01)"
                        )

            else:
                logger.warning(
                    f"Unable to automatically adjust classifier for {model_name}. "
                    "Model architecture not recognized for class count adaptation."
                )

        pretrained_model = pretrained_model.to(device)
        logger.info(f"Model moved to device: {device}")

        return pretrained_model
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        return None


def save_model_weights(model, output_dir, model_name, num_classes):
    """
    Save model weights to the specified directory
    """
    if model is None:
        logger.error("Cannot save weights: model is None")
        return False

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    weights_filename = f"{model_name}_{num_classes}classes.pth.tar"
    weights_path = Path(output_dir) / weights_filename

    try:
        torch.save(model.cpu().state_dict(), str(weights_path))
        logger.info(f"Saved model weights to {weights_path}")
        model.to(device)
        return True
    except Exception as e:
        logger.error(f"Error saving model weights: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and save pretrained model weights")

    # Get all available torchvision models
    available_models = tv_models.list_models()

    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        choices=available_models,
                        help="Name of the model from torchvision.models")
    parser.add_argument("--num_classes",
                        type=int,
                        default=1000,
                        help="Number of output classes (default: 1000)")
    parser.add_argument("--output_dir",
                        type=str,
                        default=str(
                            (Path(__file__).resolve().parent.parent.parent /
                             "pretrained/imagenet").absolute()),
                        help="Directory to save pretrained weights")

    args = parser.parse_args()

    # Get the pretrained model
    model = get_torchvision_pretrained_model(args.model_name, args.num_classes)

    if model is None:
        logger.error(f"Failed to obtain model {args.model_name}")
        return 1

    # Save the model weights
    output_dir = Path(args.output_dir)
    success = save_model_weights(model, output_dir, args.model_name,
                                 args.num_classes)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
