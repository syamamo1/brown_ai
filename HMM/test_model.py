from touchscreen_helpers.simulation_evaluator import touchscreenEvaluator
from touchscreen_helpers.simulator import touchscreenSimulator
from touchscreen import touchscreenHMM
import time

"""
You should use this to test your model! Simply run python3 test_model.py on
a department machine.
    SPHERICAL SCORE: This is your proper score, as described in the handout.pdf
    ESTIMATED GRADE: This is only an estimate. The score is normalized by the best TA solution's
        spherical score which is 0.38. Again, this is only an estimate and NOT your true grade.
    TIME: You should aim to have your model run under 60seconds on a department machine.
"""

start = time.time()
scores = []

# Run n simulations each with 100 observation frames 
n = 10
for i in range(n):
    model = touchscreenHMM()
    sim = touchscreenSimulator(width=20, height=20, frames=10000)
    sim.run_simulation() # generate the simulation data
    evaluator = touchscreenEvaluator()
    score = evaluator.evaluate_touchscreen_hmm(model, sim)
    scores.append(score)

# Report results 
score = round(sum(scores)/len(scores), 2)
print("Your average spherical score is: ", score)
print("Your estimated grade is: ", round((score/0.38), 2))
print("Your model's time to run: ", round((time.time() - start), 2))
