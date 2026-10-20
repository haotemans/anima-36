from anima36.schedule import attention_kind, build_attention_schedule

def test_canonical_schedule_counts():
    schedule = build_attention_schedule(36)
    assert len(schedule) == 36
    assert schedule.count("ls_linear") == 27
    assert schedule.count("standard") == 9

def test_every_fourth_block_is_standard():
    schedule = build_attention_schedule(36)
    assert all(schedule[i] == "standard" for i in range(3, 36, 4))
    assert all(schedule[i] == "ls_linear" for i in range(36) if i % 4 != 3)

def test_attention_kind_bounds():
    assert attention_kind(0) == "ls_linear"
    assert attention_kind(3) == "standard"
