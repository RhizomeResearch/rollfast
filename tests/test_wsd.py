import jax.numpy as jnp
from rollfast.schedules.wsd import wsd_schedule

def test_wsd_schedule():
    sched = wsd_schedule(peak_lr=0.01, total_steps=100, warmup_fraction=0.1, decay_fraction=0.1)
    lr_0 = sched(0)
    lr_50 = sched(50)
    lr_99 = sched(99)
    assert lr_0 < 0.01
    assert lr_50 == 0.01
    assert lr_99 < 0.01
