import model.enums

from batch.profile import BatchProfile

params = {
    "testing": BatchProfile(
        parent_params={
            "num_agents": 10,
            "learning_rate": [0.01, 0.05, 0.1],
            "a_b_split": [0.8, 0.6, 0.4, 0.2],
            "base_contaminated": [0.8, 0.2, 0.0],
            "ambiguity_prob": [[0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.2, 0.2]],
        },
        child_params={},
    )
}
