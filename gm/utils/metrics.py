import torch


def masked_nll(logits, labels, mask):
    """
    logits:  (B, T, V) – raw outputs
    labels:  (B, T)    – target token ids
    mask  :  (B, T)    – 1 for real tokens, 0 for padding
    return: scalar – mean NLL over unmasked tokens
    """
    vocab = logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nll = loss_fct(logits.view(-1, vocab), labels.view(-1))
    nll = nll.view_as(labels)

    nll = (nll * mask).sum() / mask.sum()
    return nll