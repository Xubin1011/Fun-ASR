from itertools import groupby

import torch
import torchaudio.functional as F


def forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0):
    items = []
    try:
        # The current version only supports batch_size==1.
        log_probs, targets = log_probs.unsqueeze(0).cpu(), targets.unsqueeze(0).cpu()
        assert log_probs.shape[1] >= targets.shape[1]
        alignments, scores = F.forced_align(log_probs, targets, blank=blank)
        alignments, scores = alignments[0], torch.exp(scores[0]).tolist()
        # use enumerate to keep track of the original indices, then group by token value
        for token, group in groupby(enumerate(alignments), key=lambda item: item[1]):
            if token == blank:
                continue
            group = list(group)
            start = group[0][0]
            end = start + len(group)
            score = max(scores[start:end])
            items.append(
                {
                    "token": token.item(),
                    "start_time": start,
                    "end_time": end,
                    "score": round(score, 3),
                }
            )
    except:
        pass
    return items
