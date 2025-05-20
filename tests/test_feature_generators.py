from super_segment.data_generation import (
    generate_age,
    generate_balance,
    generate_num_accounts,
    generate_last_login_days,
    generate_satisfaction_score,
    generate_profession,
    generate_phase,
    generate_gender,
    generate_region,
    generate_risk_profile,
    generate_contrib_freq,
    generate_logins_per_month,
)


def test_feature_generators(config):
    assert 25 <= generate_age(config) <= 65
    assert 20000 <= generate_balance(config) <= 300000
    assert generate_num_accounts(config) in [1, 2]
    assert 1 <= generate_last_login_days(config) <= 180
    assert generate_satisfaction_score(config) in [1, 2, 3]
    assert generate_profession(config) in ["A", "B"]
    assert generate_phase(config) in ["Accumulation", "Retirement"]
    assert generate_gender(config) in ["Male", "Female"]
    assert generate_region(config) in ["NSW", "VIC"]
    assert generate_risk_profile(config) in ["Conservative", "Aggressive"]
    assert generate_contrib_freq(config) in ["Monthly", "Yearly"]
    assert 0 <= generate_logins_per_month(config) <= 10
