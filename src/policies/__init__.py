from policies.dice import DICEPolicy
from policies.fair import FairDICEPolicy, FairRacingPolicy, FairRiskOnlyPolicy
from policies.racing import PairedRacingPolicy
from policies.risk_only import RiskOnlyPolicy
from policies.uniform import UniformPolicy

POLICY_MAP = {
    "uniform": UniformPolicy,
    "racing": PairedRacingPolicy,
    "risk_only": RiskOnlyPolicy,
    "dice": DICEPolicy,
    # Post-hoc diagnostic only (equal early-stop + Uniform fallback):
    "fair_racing": FairRacingPolicy,
    "fair_dice": FairDICEPolicy,
    "fair_risk_only": FairRiskOnlyPolicy,
}


def get_policy(name: str):
    if name not in POLICY_MAP:
        raise KeyError(name)
    return POLICY_MAP[name]()
