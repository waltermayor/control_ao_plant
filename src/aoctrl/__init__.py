"""aoctrl — adaptive-optics control research framework.

Layers (low -> high), each depending only on the ones below it:

    utils        device, seeding, caching, paths
    optics       physical simulation backends (OOPAO, synthetic) behind one interface
    plant        control-facing plant: step(u) -> y, delay, saturation, operating conditions
    calibration  modal basis, interaction matrix, reconstructor, OPD -> modal projection
    dynamics     turbulence identification (AR models, prediction)
    sensors      WFS models (optical gain), noise characterisation
    estimators   state estimators (EKF / KF)
    control      controllers + LQR/DARE solvers
    models       learned latent models (PyTorch) + numpy inference ops
    training     generic trainer, checkpoints
    simulation   closed-loop runner and telemetry
    metrics      performance / modal / estimation metrics
    evaluation   sweeps over conditions, summaries, diagnostics
    viz          plots and videos
    tracking     W&B + local logging
    experiments  runnable experiments composed from the above
"""

__version__ = "0.1.0"
