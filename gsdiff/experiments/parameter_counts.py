"""Pure optimizer-membership parameter-count formulas for locked methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gsdiff.data._artifact_dataset import _validate_blind_acquisition_spec

from .methods import ResolvedMethod


def _positive_int(value: object, noun: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{noun} must be a positive integer")
    return value


def _motion_parameter_count(config: Mapping[str, object]) -> int:
    rotation = config.get("enable_rotation")
    degree = config.get("polynomial_degree")
    affine = config.get("enable_affine")
    if type(rotation) is not bool or degree not in (1, 2) or type(affine) is not bool:
        raise ValueError("locked motion parameter contract is invalid")
    rotational = int(rotation)
    return (
        2
        + rotational
        + (2 + rotational if degree == 2 else 0)
        + (3 if affine else 0)
    )


def _gidc_parameter_count(channels: object) -> int:
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
        raise ValueError("locked GIDC channels are invalid")
    values = [_positive_int(value, "GIDC channel") for value in channels]
    if len(values) != 5:
        raise ValueError("locked GIDC requires five channel widths")
    c0, _c1, _c2, _c3, _c4 = values

    def block(input_channels: int, output_channels: int) -> int:
        return 25 * input_channels * output_channels + 2 * output_channels

    def down(input_channels: int, output_channels: int) -> int:
        return (
            25 * input_channels * output_channels
            + 25 * output_channels * output_channels
            + 4 * output_channels
        )

    def up(lower_channels: int, upper_channels: int) -> int:
        return (
            16 * lower_channels * upper_channels
            + 75 * upper_channels * upper_channels
            + 4 * upper_channels
        )

    total = block(2, c0) + block(c0, c0)
    total += sum(down(values[index], values[index + 1]) for index in range(4))
    total += sum(
        up(values[index + 1], values[index]) for index in range(3, -1, -1)
    )
    return total + 9 * c0 + 2


def _recinr_parameter_count(
    representation: Mapping[str, object],
    h: int,
    w: int,
) -> int:
    if representation.get("basis") != "lowrank":
        raise ValueError("locked ReCINR parameter formula supports lowrank only")
    hidden = _positive_int(representation.get("hidden_dim"), "ReCINR hidden width")
    layers = _positive_int(
        representation.get("render_layers"),
        "ReCINR render layer count",
    )
    order = representation.get("basis_order")
    harmonics = representation.get("harmonics")
    if type(order) is not int or order not in {0, 1, 2, 3}:
        raise ValueError("locked ReCINR basis order is invalid")
    if type(harmonics) is not int or harmonics < 0:
        raise ValueError("locked ReCINR harmonics are invalid")
    basis_count = {0: 5, 1: 3, 2: 6, 3: 8}[order]
    temporal_width = 1 + 2 * harmonics
    canonical = hidden * h * w
    flow_scale = 1
    warp = (
        temporal_width * hidden
        + hidden
        + hidden * hidden
        + hidden
        + hidden * (2 * basis_count)
        + 2 * basis_count
    )
    renderer = max(1, layers) * (hidden * hidden + hidden) + hidden + 1
    return canonical + flow_scale + warp + renderer


def _recinr_se2_scene_parameter_count(
    scene: Mapping[str, object],
    h: int,
    w: int,
) -> int:
    channels = _positive_int(scene.get("channels"), "ReCINR-SE2 channels")
    layers = _positive_int(
        scene.get("render_layers"),
        "ReCINR-SE2 render layer count",
    )
    grid_size = _positive_int(
        scene.get("grid_size"),
        "ReCINR-SE2 grid size",
    )
    scale = grid_size / min(h, w)
    grid_h, grid_w = round(h * scale), round(w * scale)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("ReCINR-SE2 scaled grid is invalid")
    return (
        channels * grid_h * grid_w
        + max(1, layers) * (channels * channels + channels)
        + channels
        + 1
    )


def expected_trainable_parameter_count(
    method: ResolvedMethod,
    expected_acquisition_spec: Mapping[str, object],
) -> int:
    """Count the max unique active optimizer-owned trainable tensors."""
    if type(method) is not ResolvedMethod:
        raise TypeError("method must be a ResolvedMethod")
    spec = _validate_blind_acquisition_spec(expected_acquisition_spec)
    dimensions = spec["dimensions"]
    assert isinstance(dimensions, Mapping)
    h, w, t = (
        int(dimensions["H"]),
        int(dimensions["W"]),
        int(dimensions["T"]),
    )
    return _expected_trainable_parameter_count_for_dimensions(
        method,
        h=h,
        w=w,
        t=t,
    )


def _expected_trainable_parameter_count_for_dimensions(
    method: ResolvedMethod,
    *,
    h: int,
    w: int,
    t: int,
) -> int:
    for name, value in (("H", h), ("W", w), ("T", t)):
        _positive_int(value, name)
    if method.method_id == "dgi":
        return 0
    if method.method_id == "static_cs":
        return h * w
    if method.method_id in {"perframe_cs", "tv3d"}:
        return t * h * w
    if method.method_id == "monin":
        solver = method.semantic_config["solver"]
        assert isinstance(solver, Mapping)
        degree = solver["polynomial_degree"]
        if type(degree) is not int or degree < 0:
            raise ValueError("locked polynomial degree is invalid")
        return h * w + 2 * degree
    if method.method_id == "gidc3dtv":
        solver = method.semantic_config["solver"]
        assert isinstance(solver, Mapping)
        return _gidc_parameter_count(solver["unet_channels"])
    if method.method_id == "recinr":
        representation = method.semantic_config["representation"]
        assert isinstance(representation, Mapping)
        return _recinr_parameter_count(representation, h, w)

    scene = method.semantic_config["scene"]
    motion = method.semantic_config["motion"]
    assert isinstance(scene, Mapping) and isinstance(motion, Mapping)
    motion_count = _motion_parameter_count(motion)
    if method.method_id == "siren":
        hidden = _positive_int(scene.get("hidden"), "SIREN hidden width")
        layers = _positive_int(scene.get("hidden_layers"), "SIREN hidden layers")
        scene_count = layers * hidden * hidden + (layers + 4) * hidden + 1
    elif method.method_id == "recinr_se2":
        scene_count = _recinr_se2_scene_parameter_count(scene, h, w)
    elif method.method_id in {"gsdiff_tv", "gsdiff_diffusion"}:
        gaussian_count = _positive_int(
            scene.get("gaussian_count"),
            "Gaussian scene count",
        )
        scene_count = 6 * gaussian_count
    else:
        raise ValueError("unknown method parameter-count contract")
    return scene_count + motion_count
