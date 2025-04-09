import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def calculate_similarity_scores(items):
    num_slices = items.shape[0]
    similarity_scores = []
    for i in range(num_slices):
        current_slice = items[i].flatten().reshape(1, -1)
        other_slices = np.delete(items, i, axis=0).reshape(-1, items.shape[1] * items.shape[2])
        similarities = cosine_similarity(current_slice, other_slices)
        score = similarities.sum()
        similarity_scores.append(score)
    return np.array(similarity_scores)


# todo : rank the input
def rank_slices(slices):
    similarity_scores = calculate_similarity_scores(slices)
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_slices = slices[ranked_indices]
    return ranked_slices


def rank_cond_frames(cond_frame_outputs):
    features = []
    for cond_frame_index in cond_frame_outputs:
        features.append(cond_frame_outputs[cond_frame_index]["maskmem_features"])
    similarity_scores = calculate_similarity_scores(np.array(features))
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_cond_frame_outputs = cond_frame_outputs[ranked_indices]
    return ranked_cond_frame_outputs


def select_similar_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are similar to the current frame at `frame_idx`. Here, we take
    - the similar conditioning frame `frame_idx` (if any);

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        ranked_cond_frame_outputs = rank_cond_frames(cond_frame_outputs)
        for cond_frame_index in ranked_cond_frame_outputs:
            selected_outputs[cond_frame_index] = cond_frame_outputs[cond_frame_index]

        # add the similar conditioning frame until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs
