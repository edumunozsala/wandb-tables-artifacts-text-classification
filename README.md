# Text Classification on User Complaints: Using Weights & Biases for tracking and evaluation

## Description 

This repo contains notebooks and python files to introduce you in the use of W&B Tables and Artifacts to make experiments, register models and metrics and finally evaluate the performance of a model. The model is very simple, it is not the focus of this project, we are only interested on hot to integrate your training and hyperparameter tuning stages with W&B to keep track of your experiments and results.

After executing all the steps, we built some W&B reports with results, you can check them in:
- [Baseline Report: Text Classification on user complaints](https://wandb.ai/edumunozsala/consumer_complaints_classification/reports/Baseline-Report-Text-Classification-on-user-complaints--Vmlldzo1MzI2NjEw)
- [Hyperparameter Tuning for a simple CNN 1D Text Classifier](https://wandb.ai/edumunozsala/consumer_complaints_classification/reports/Hyperparameter-Tuning-for-a-simple-CNN-1D-Text-Classifier--Vmlldzo1NDkxMTEz)
- [Model Evaluation and Error Analysis](https://api.wandb.ai/links/edumunozsala/t6j1bzog)

## Content
There are a bunch of notebooks and python files in this repo:
- `user_complaints_wandb_data_collect`: in this repo we analyze the dataset and process it for training.
- `user_complaints_wandb_baseline`: create a baseline model and evaluate its performance, it will be our baseline model.
- `user_complaints_wandb_refactor_code`: here we refactor our training code to be use for hyperparameter tuning with sweeps.
- `train.py` `params.py` `sweep.yaml`: code and config files to execute a hyperparameter tuning job.
- `user_complaints_category_wandb_evaluation`: final evluation of our tuned model.
- `user_complaints_wandb_EDA`: this notebook contains code to do an EDA of the text data, some pieces of code are reused in the data collect notebook.


## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a Apache 2.0 license.

