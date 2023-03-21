import argparse
import json

from loaders import CSVLoader
from processor import Processor
from sampler import Sampler
from trainers import Trainer

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--path", type=str, default="./models")
parser.add_argument("--handle-missing", action="store_true")
parser.add_argument("--optimize", action="store_true")
parser.add_argument("--model-args", type=str, default="{}")
args = parser.parse_args()

# Loads the data
train = CSVLoader("./data/train.csv").load()

# Process the data
X = Processor(train).process()
y = train["Survived"]

# Split in samples
X_train, X_test, y_train, y_test = Sampler(X, y).split()

# Setup trainer
trainer = Trainer(
    args.model,
    handle_missing=args.handle_missing,
    model_args=json.loads(args.model_args),
)

# Fits the model
trainer.train(X_train, y_train)

# Check if runs grid search
if args.optimize:
    trainer.optimize()

# Show results
print(
    f"Trained {trainer.model_name} with hyperparameters:",
    trainer.model.model.get_params(),
)
print(f"\nAccuracy Score for {args.model}:", trainer.evaluate(X_test, y_test))

# Save trained model
trainer.save(args.path, args.model)
