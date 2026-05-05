from lerobot.processor.pipeline import PolicyProcessorPipeline


def make_molmoact2_pre_post_processors(config=None, dataset_stats=None):
    """MolmoAct2 performs normalization and action scaling inside the model."""
    del config, dataset_stats

    def identity(x):
        return x

    return (
        PolicyProcessorPipeline(
            name="molmoact2_pre",
            steps=[],
            to_transition=identity,
            to_output=identity,
        ),
        PolicyProcessorPipeline(
            name="molmoact2_post",
            steps=[],
            to_transition=identity,
            to_output=identity,
        ),
    )
