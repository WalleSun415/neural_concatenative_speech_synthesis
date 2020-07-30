import torch


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def to_gpu(x):
    # x = x.contiguous()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return x.to(device)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    # ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    ids = to_gpu(ids)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask