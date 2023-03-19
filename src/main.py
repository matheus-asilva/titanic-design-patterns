import argparse

from loaders import CSVLoader
from processor import Processor
from sampler import Sampler
from trainers import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="decision_tree")
parser.add_argument("--path", type=str, default="./models")
parser.add_argument("--handle_missing", action="store_true")
parser.add_argument("--optimize", action="store_true")
args = parser.parse_args()

train = CSVLoader("./data/train.csv").load()

if args.model not in ["xgboost", "lightgbm"]:
    args.handle_missing = True

X = Processor(train).process(handle_missing=args.handle_missing)
y = train["Survived"]

X_train, X_test, y_train, y_test = Sampler(X, y).split()

trainer = Trainer(args.model)
trainer.train(X_train, y_train)

if args.optimize:
    trainer.optimize()

print(f"Accuracy Score for {args.model}:", trainer.evaluate(X_test, y_test))
trainer.save(args.path, args.model)
