"""M37: the file-depth cut declared before any run, and the rows recorded under it beside the committed chunk-cut rows."""

from app.eval import file_depth


def _path(cid: str) -> str:
    return cid.split("#")[0]


def test_the_rule_is_declared_as_the_owner_worded_it():
    assert file_depth.FILE_DEPTH_RULE == (
        "file-depth cut: every list (the BM25 leg, the dense leg and the fused list) keeps chunks in rank order until it "
        "holds retrieve_k distinct files (or its candidates run out); metrics are unchanged.")


def test_the_cut_keeps_chunks_in_rank_order_until_the_list_holds_retrieve_k_files():
    ranked = ["a#0", "a#1", "b#0", "c#0", "b#1", "d#0", "a#2"]
    assert file_depth.file_depth_cut(ranked, _path, 3) == ["a#0", "a#1", "b#0", "c#0"]  # stops at the third file
    assert file_depth.file_depth_cut(ranked, _path, 1) == ["a#0"]
    assert file_depth.file_depth_cut(ranked, _path, 4) == ["a#0", "a#1", "b#0", "c#0", "b#1", "d#0"]


def test_a_list_with_fewer_files_than_retrieve_k_keeps_every_candidate():
    ranked = ["a#0", "b#0", "a#1"]
    assert file_depth.file_depth_cut(ranked, _path, 50) == ranked
    assert file_depth.file_depth_cut([], _path, 50) == []


def test_a_chunk_cut_list_can_hold_fewer_files_than_the_file_depth_list():
    # the reason for the rule: 4 chunks hold 2 files under a 4-chunk cut, while the file-depth list reaches 4 files
    ranked = ["a#0", "a#1", "b#0", "b#1", "c#0", "d#0"]
    chunk_cut = ranked[:4]
    assert len({_path(c) for c in chunk_cut}) == 2
    assert len({_path(c) for c in file_depth.file_depth_cut(ranked, _path, 4)}) == 4
