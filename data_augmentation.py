import Augmentor

p = Augmentor.Pipeline("data/augment/inputs")
p.ground_truth("data/augment/targets")

p.zoom_random(probability=0.5, percentage_area=0.9)
p.sample(10)
