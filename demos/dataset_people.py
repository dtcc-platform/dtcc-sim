"""Animate synthetic pedestrians on mapped streets around Chalmers."""

from dtcc_sim import datasets

bounds = (319845, 6397785, 320095, 6398035)

people = datasets.people(bounds=bounds, profile="walking_preview")
people.info()
people.plot(animate=True)
