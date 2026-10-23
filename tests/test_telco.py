from __future__ import annotations

from churnguard.modeling.telco import NUMERIC, load


def test_blank_total_charges_become_missing(tmp_path):
    csv = tmp_path / "telco.csv"
    csv.write_text(
        "customerID,gender,SeniorCitizen,tenure,Contract,MonthlyCharges,TotalCharges,Churn\n"
        "a,Male,0,0,Month-to-month,20.5, ,No\n"
        "b,Female,1,12,One year,70.0,840.0,Yes\n")
    X, y, num, cat = load(csv)
    assert y.tolist() == [0, 1]
    assert X["TotalCharges"].isna().tolist() == [True, False]
    assert set(num) == set(NUMERIC) and set(cat) == {"gender", "Contract"}
    assert "customerID" not in X
