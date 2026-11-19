import pandas as pd

from fraud.data import load_transactions, tokenize_cards


def test_token_is_stable_hides_pan_and_depends_on_key(monkeypatch):
    s = pd.Series(["4111111111111111", "4222222222222222", "4111111111111111"])
    a = tokenize_cards(s)
    assert a[0] == a[2] != a[1]
    assert not a.str.contains("4111").any() and a.str.len().eq(16).all()
    monkeypatch.setenv("FRAUD_CARD_KEY", "another-key")
    assert tokenize_cards(s)[0] != a[0]


def test_loaded_frame_never_exposes_the_card_number(csv_paths):
    frame = load_transactions(csv_paths[0])
    assert frame["cc_num"].str.len().eq(16).all()  # synthetic PANs were 1-2 digits
