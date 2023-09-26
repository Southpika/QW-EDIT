import os
import argparse
import json
from typing import Optional

from .rome import ROMEHyperParams, apply_rome_to_model
from .utils.template import data_transform
from .utils.loader import load_model_and_tokenizer
from .utils.context import data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        default=data,
                        help = 'data path or list of dictionary, contains keys:subject,target')

    parser.add_argument('--config',
                        default = 'qwen',
                        help = 'config model name')

    parser.add_argument('--template',
                        default = 'qwen',
                        help = 'template model name')

    parser.add_argument('--output',
                        default = None,
                        help = 'output dir')
          
    config = parser.parse_args()
    return config


args = get_parser()

def rome_edit(
    data: str, model: str, config: str, template: Optional[str] = "default",
    output: Optional[str] = None, checkpointing: Optional[bool] = False
) -> None:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        data (`str`):
            The path of the `json` file containing the samples for editing.
        model (`str`):
            The name or path of the pre-trained transformer model to be edited.
        config (`str`):
            The name of the hyper-parameters to use for editing the model.
        template (`str`, *optional*, defaults to `default`):
            The name of the template to use in generation.
        output (`str`, *optional*, defaults to `None`):
            The path to save the edited model.
        checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing or not.
    """
    if isinstance(data,list):
        requests = data
    else:
        assert os.path.exists(data), "data not found"

        with open(data, "r", encoding="utf-8") as f:
            requests = json.load(f)

    requests = [data_transform(request,template=template) for request in requests]
    model_old, tokenizer, batch_first = load_model_and_tokenizer(model, checkpointing)

    print("Retrieving hyperparameters..")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

    print(f"Applying rome to model..")
    model_new, _ = apply_rome_to_model(
        model_old,
        tokenizer,
        requests,
        hparams,
        batch_first,
        return_diff_weights=False
    )


    if output is not None:
        model_new.config.use_cache = True
        model_new.save_pretrained(output)
        tokenizer.save_pretrained(output)
    
    return model_new,tokenizer



if __name__ == "__main__":
    rome_edit(args.data,args.model,args.config,args.template)
