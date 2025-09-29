from spike3_chunking_and_metadata.transcript import validate_transcript_df


def test_validate_transcript_df(sample_df):
    errs = validate_transcript_df(sample_df)
    assert errs == []  # clean transcript
