"""Tests for name-variant entity resolution."""

from engram.services.entity_resolution import is_name_variant, resolve_variant


def test_first_name_resolves_to_full_name():
    assert is_name_variant("caroline", "caroline kim")
    assert is_name_variant("Albert", "Albert Villarde")


def test_initial_matches_full_token():
    assert is_name_variant("Anna B", "Anna Bethke")
    assert is_name_variant("Mike A", "Mike Alden")


def test_distinct_people_sharing_first_name_not_merged():
    assert not is_name_variant("Mark Guzman", "Mark Zschocke")
    assert not is_name_variant("Caroline", "Charlie")


def test_resolve_prefers_fullest_candidate():
    assert resolve_variant("caroline", ["caroline kim", "jay"]) == "caroline kim"


def test_resolve_returns_none_without_match():
    assert resolve_variant("caroline", ["jay", "nick"]) is None


def test_resolve_ambiguous_returns_none():
    # Two distinct fuller names sharing the first name -> do not merge.
    assert resolve_variant("mark", ["mark guzman", "mark zschocke"]) is None


def test_resolve_does_not_collapse_fuller_into_shorter():
    # A fuller mention must never be reused as a shorter existing entity.
    assert resolve_variant("caroline kim", ["caroline"]) is None
    assert resolve_variant("anna bethke", ["anna b"]) is None


def test_resolve_shorter_into_fuller():
    assert resolve_variant("caroline", ["caroline kim"]) == "caroline kim"
