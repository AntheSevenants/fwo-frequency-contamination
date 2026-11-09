import model.enums

from batch.profile import BatchProfile

params = {
    "article": BatchProfile(
        parent_params={
            "num_agents": 10,
            "learning_rate": [0.01],
            "a_b_split": [0.8, 0.6, 0.5, 0.4, 0.2],
            "base_contaminated": [0.8, 0.6, 0.5, 0.4, 0.2, 0.0],
        },
        child_params={
            "fine-grained": {
                "ambiguity_prob": [[0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.2, 0.2]],
                "construction_storage": model.enums.ConstructionStorage.FINE_GRAINED,
            },
            "coarse-grained": {
                "construction_storage": model.enums.ConstructionStorage.COARSE,
            },
        },
    )
}
