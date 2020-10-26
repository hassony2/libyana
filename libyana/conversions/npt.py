import torch


def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return tensor


def tensorify(array, device=None):
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    if device is not None:
        array = array.to(device)
    return array
