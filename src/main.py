import argparse

from loaders import CSVLoader
from processor import Processor
from sampler import Sampler
from trainers import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--path", type=str, default="./models")
parser.add_argument("--handle_missing", action="store_true")
parser.add_argument("--optimize", action="store_true")
args = parser.parse_args()

train = CSVLoader("./data/train.csv").load()

X = Processor(train).process()
y = train["Survived"]

X_train, X_test, y_train, y_test = Sampler(X, y).split()

trainer = Trainer(args.model, handle_missing=args.handle_missing)
trainer.train(X_train, y_train)

if args.optimize:
    trainer.optimize()

print(f"Accuracy Score for {args.model}:", trainer.evaluate(X_test, y_test))
trainer.save(args.path, args.model)
