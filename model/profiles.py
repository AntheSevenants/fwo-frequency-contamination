import model.enums
import model.constructions.parsing

from batch.profile import BatchProfile

params = {
    "testing": BatchProfile(
        parent_params={
            "num_agents": 10,
            "parsing_mode": [
                model.constructions.parsing.ExactParsing(),
            ],
            "a_group_share": 0.5,
            "base_contaminated": [[0.2, 0]],
            "homophily": [[0.98, 0.99]],
        },
        child_params={},
    )
}
