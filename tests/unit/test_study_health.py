from pgm_asc.experiments.study_runner_health import new_traceback


def test_detects_fatal_log_before_process_exit_and_ignores_prior_segment(tmp_path):
    path=tmp_path/'worker.log'
    path.write_text('old error\nTraceback (most recent call last):\nold\n')
    offset=path.stat().st_size
    with path.open('a') as f:f.write('new worker initialized\n')
    assert not new_traceback(path,offset)
    with path.open('a') as f:f.write('Traceback (most recent call last):\nnew fatal\n')
    assert new_traceback(path,offset)
