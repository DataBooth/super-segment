import pytest
from super_segment.model import SuperannuationSegmentationModel


def test_segmentation_model_train_and_predict(config):
    from super_segment.data_generation import MemberDataGenerator

    gen = MemberDataGenerator(config)
    df = gen.generate(n_member=20)
    model = SuperannuationSegmentationModel(n_clusters=2)
    stats = model.train(df)
    assert "silhouette" in stats
    assert "cluster_sizes" in stats
    assert model.is_trained

    # Predict segment for a sample member
    sample = df.iloc[0].to_dict()
    seg = model.predict_segment(sample)
    assert isinstance(seg, int)
    assert 0 <= seg < model.n_clusters


def test_segmentation_model_add_segments(config):
    from super_segment.data_generation import MemberDataGenerator

    gen = MemberDataGenerator(config)
    df = gen.generate(n_member=10)
    model = SuperannuationSegmentationModel(n_clusters=2)
    model.train(df)
    df_with_segments = model.add_segments(df)
    assert "segment" in df_with_segments.columns
    assert set(df_with_segments["segment"].unique()).issubset(
        set(range(model.n_clusters))
    )
