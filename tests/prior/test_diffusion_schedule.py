import pytest
import torch

import gsdiff.prior.diffusion as diffusion_module
from gsdiff.prior.diffusion import DiffusionPrior, log_annealed_sigma
from train import _record_sigma_used


class ZeroDenoiser(torch.nn.Module):
    def forward(self, x, sigma):
        return torch.zeros_like(x)


def make_cpu_prior(monkeypatch, *, denoise_steps=1, ddim_spacing="linear"):
    denoiser = ZeroDenoiser()
    monkeypatch.setattr(diffusion_module, "UNet3D", lambda **kwargs: denoiser)
    monkeypatch.setattr(diffusion_module.torch, "load", lambda *args, **kwargs: {})
    prior = DiffusionPrior(
        checkpoint_path="unused.pt",
        device="cpu",
        denoise_steps=denoise_steps,
        sigma_min=0.002,
        sigma_max=0.5,
        sigma_start=0.3,
        sigma_end=0.05,
        ddim_spacing=ddim_spacing,
    )
    return prior


def test_one_step_schedule_uses_end_sigma():
    assert log_annealed_sigma(0, 1, 0.3, 0.05) == 0.05


def test_multistep_schedule_hits_requested_endpoints():
    values = [log_annealed_sigma(i, 5, 0.3, 0.05) for i in range(5)]

    assert values[0] == 0.3
    assert values[-1] == 0.05
    assert all(a > b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("count", [0, -1])
def test_schedule_rejects_nonpositive_count(count):
    with pytest.raises(ValueError, match="count must be >= 1"):
        log_annealed_sigma(0, count, 0.3, 0.05)


@pytest.mark.parametrize("index", [-1, 4])
def test_schedule_rejects_index_outside_call_range(index):
    with pytest.raises(IndexError, match="outside"):
        log_annealed_sigma(index, 4, 0.3, 0.05)


@pytest.mark.parametrize(
    ("sigma_start", "sigma_end"),
    [(0.0, 0.05), (-0.3, 0.05), (0.3, 0.0), (0.3, -0.05)],
)
def test_schedule_rejects_nonpositive_sigma_values(sigma_start, sigma_end):
    with pytest.raises(ValueError, match="sigma_start and sigma_end must be positive"):
        log_annealed_sigma(0, 4, sigma_start, sigma_end)


def test_prior_current_sigma_uses_exact_outer_schedule():
    prior = DiffusionPrior.__new__(DiffusionPrior)
    prior.sigma_start = 0.3
    prior.sigma_end = 0.05
    prior._n_steps = 4
    prior._call_count = 0

    seen = []
    for _ in range(4):
        seen.append(prior._current_sigma())
        prior._call_count += 1

    assert seen == pytest.approx([0.3, 0.16509636, 0.09085603, 0.05])


def test_prior_initial_last_sigma_is_none_and_read_only(monkeypatch):
    prior = make_cpu_prior(monkeypatch)

    assert prior.last_sigma is None
    with pytest.raises(AttributeError):
        prior.last_sigma = 0.1


def test_set_n_steps_rejects_zero(monkeypatch):
    prior = make_cpu_prior(monkeypatch)

    with pytest.raises(ValueError, match="n must be >= 1"):
        prior.set_n_steps(0)


def test_set_n_steps_resets_call_accounting_and_last_sigma(monkeypatch):
    prior = make_cpu_prior(monkeypatch)
    prior.set_n_steps(1)
    prior.proximal(torch.zeros(2, 1, 3, 4), weight=0.0)

    prior.set_n_steps(4)

    assert prior._call_count == 0
    assert prior.last_sigma is None


def test_one_proximal_call_consumes_end_sigma(monkeypatch):
    prior = make_cpu_prior(monkeypatch)
    prior.set_n_steps(1)

    output = prior.proximal(torch.zeros(2, 1, 3, 4), weight=0.0)

    assert torch.equal(output, torch.zeros_like(output))
    assert prior.last_sigma == 0.05


def test_four_proximal_calls_consume_exact_outer_schedule(monkeypatch):
    prior = make_cpu_prior(monkeypatch)
    prior.set_n_steps(4)
    x = torch.zeros(2, 1, 3, 4)

    seen = []
    for _ in range(4):
        prior.proximal(x, weight=0.0)
        seen.append(prior.last_sigma)

    assert seen == pytest.approx([0.3, 0.16509636, 0.09085603, 0.05])


def test_proximal_records_sigma_before_incrementing_call_count(monkeypatch):
    prior = make_cpu_prior(monkeypatch)
    prior.set_n_steps(4)
    observations = []

    class ObservedCount(int):
        def __add__(self, increment):
            observations.append((int(self), prior.last_sigma))
            return int(self) + increment

    prior._call_count = ObservedCount(0)
    prior.proximal(torch.zeros(2, 1, 3, 4), weight=0.0)

    assert observations == [(0, 0.3)]


@pytest.mark.parametrize("ddim_spacing", ["linear", "log"])
def test_internal_ddim_ladder_includes_requested_endpoints(
    monkeypatch, ddim_spacing
):
    prior = make_cpu_prior(
        monkeypatch, denoise_steps=3, ddim_spacing=ddim_spacing
    )

    sigmas = prior._ddim_sigma_ladder(0.3)

    assert sigmas[0].item() == pytest.approx(0.3)
    assert sigmas[-1].item() == pytest.approx(0.002)
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_training_history_records_consumed_sigma_without_schedule_lookahead():
    class CompletedPrior:
        last_sigma = 0.05

        def _current_sigma(self):
            raise AssertionError("completed iterations must not query the next sigma")

    info = {"loss_data": 1.0}

    sigma_used = _record_sigma_used(info, CompletedPrior())

    assert sigma_used == 0.05
    assert info == {"loss_data": 1.0, "sigma_used": 0.05}
