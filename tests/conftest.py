"""Keep historical terminal integration scripts out of offline collection.

The source workspace retains originals; the public release archives them.
"""
collect_ignore = [
    "test_bridge_signals.py",
    "test_dll_protocol.py",
    "test_signals_near_end.py",
    "test_single_job_match_port.py",
    "test_terminal_manager_basic.py",
]
