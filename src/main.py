import argparse
from loaders import CSVLoader
from trainers import Trainer
from processor import Processor
from sampler import Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="decision_tree")
parser.add_argument("--path", type=str, default="./models")
args = parser.parse_args()

train = CSVLoader("./data/train.csv").load()

X = Processor(train).process()
y = train["Survived"]

X_train, X_test, y_train, y_test = Sampler(X, y).split()

trainer = Trainer(args.model)
trainer.train(X_train, y_train)
print(f"Accuracy Score for {args.model}:", trainer.evaluate(X_test, y_test))
trainer.save(args.path, args.model)
