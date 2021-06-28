import torch
from collections import OrderedDict


def restore_pretrained_backend(
    backend: torch.nn.Module, checkpoint_path: str, debug: bool = False
) -> torch.nn.Module:
    """
    Converts the state dict of encoder of Hybrid 2 model such that it matches with the state dict
    of backend(features) model of WrapperModel class.
    NOTE: Make sure the underlying Resnet architecture is same in both the models.
    Args:
        backend (torch.nn.Module): initialized backend model.
        checkpoint_path (str): path to the saved hybrid 2 model.
        debug (bool):Prints the kys that are matched. Default set to False.
    Returns
        (torch.nn.Module): ipdated backend model with weights from hybrid 2.
    """
    pretrained_state_dict = torch.load(checkpoint_path)
    backend_state_dict = backend.state_dict()
    matched_key_pairs = []
    updated_elems = []
    while len(backend_state_dict):
        backend_elem = backend_state_dict.popitem(last=False)
        pretrained_elem = pretrained_state_dict.popitem(last=False)
        updated_elems += [(backend_elem[0], pretrained_elem[1])]
        matched_key_pairs += [(backend_elem[0], pretrained_elem[0])]
    if debug:
        print(f"These keys are matched :\n{matched_key_pairs}")
    updated_backend_dict = OrderedDict(updated_elems)
    backend.load_state_dict(updated_backend_dict)
    return backend
