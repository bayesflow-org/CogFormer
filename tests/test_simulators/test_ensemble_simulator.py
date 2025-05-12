def test_ensemble_simulator(ensemble_simulator):
    results = ensemble_simulator.run(x=[1, 2], y=[10, 20])
    assert "adder" in results
    assert results["adder"][0]["sum"] == 11
    assert results["adder"][1]["sum"] == 22
