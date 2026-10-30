# Tester configuration

Machine-specific `.ini` files are generated locally and excluded from publication.
The [example](examples/model4_orb.ini) shows the required real-tick model setting;
it is a template, not a configured broker login.

Read [MT5 setup](../docs/MT5_SETUP.md) before running a tester batch. Match symbol
specifications, timezone, warmup, costs and EA inputs to the frozen study. An `.ini`
with `Model=0` tests generated ticks; `Model=4` requests imported real ticks.
