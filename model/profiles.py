import model.enums
import model.constructions.parsing

from batch.profile import BatchProfile

params = {
    "testing": BatchProfile(
        parent_params={
            "num_agents": 10,
            "parsing_mode": [
                model.constructions.parsing.ExactParsing(),
                model.constructions.parsing.LazyParsing(0),
                model.constructions.parsing.LazyParsing(1),
                model.constructions.parsing.LazyParsing(10),
            ],
        },
        child_params={},
    )
}
