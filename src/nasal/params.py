from dataclasses import asdict, dataclass

SIDES = ("left", "right")
REGIONS = ("vestibule", "valve", "turbinate", "choana")


@dataclass(frozen=True)
class ParamSpec:
    low: float
    high: float
    unit: str
    description: str


# Single source of truth for case parameters. The generator, the Critic's
# out-of-range check, and the README table are all derived from this.
PARAM_SPECS: dict[str, ParamSpec] = {
    "valve_angle_deg": ParamSpec(
        8.0, 20.0, "deg", "Internal nasal valve angle (both sides)"
    ),
    "turbinate_scale": ParamSpec(
        0.6, 1.6, "-", "Inferior turbinate size; 1.0 is the nominal size"
    ),
    "septum_deviation": ParamSpec(
        -1.0, 1.0, "-", "Septum bend; positive narrows the left side, negative the right"
    ),
}


@dataclass(frozen=True)
class CaseParams:
    valve_angle_deg: float = 15.0
    turbinate_scale: float = 1.0
    septum_deviation: float = 0.0

    def out_of_range(self) -> list[str]:
        values = asdict(self)
        return [
            name
            for name, spec in PARAM_SPECS.items()
            if not spec.low <= values[name] <= spec.high
        ]

    def validate(self) -> "CaseParams":
        bad = self.out_of_range()
        if bad:
            raise ValueError(f"parameters out of range: {', '.join(bad)}")
        return self
