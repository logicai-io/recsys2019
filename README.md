# recsys2019

This is the code for the 1st place ACM Recsys competition.

If you have any questions please write to pawel@[address of our company website]

Team members
=================

- Paweł Jankiewicz (https://www.linkedin.com/in/pjankiewicz/)
- Liudmyla Kyrashchuk (https://www.linkedin.com/in/liudmylakyrashchuk/)
- Paweł Sienkowski (https://www.linkedin.com/in/pawel-sienkowski/)
- Magdalena Wójcik (https://www.linkedin.com/in/claygirl/)

Best single model
=================

Below there is a process to generate best single model.

Current process (full)
----------------------

1. cd data/
2. ./download_data.sh
3. cd ../src/recsys/data_prep
4. ./run_data_prep.sh
5. C++ feature generation
```
cd ../../../cpp
make
./build/price # extracts price features
./build/scores # extracts incremental features for each impression (|-separated format)
./build/extractor # extracts comparison features for |-separated features (extracted in scores)
python select_features.py
```
6. cd ../src/recsys/data_generator/
7. cd data_generator; python generate_data_parallel_all.py; cd .. (pypy is good)
8. python split_events_sorted_trans.py
9. python vectorize_datasets.py
10. python model_val.py (validate model)
11. python model_submit.py (make test predictions)
12. python make_blend.py (prepare submission file)

Quick validation (1 million rows)
----------------------

1. cd data/
2. ./download_data.sh
3. cd ../../src/recsys/data_prep
4. ./run_data_prep.sh
5. cd ..
6. cd data_generator; python generate_data_parallel_quick.py; cd - (pypy is good)
7. python quick_validate.py

Blend
=====

To create the blend we merged 37 LGBMLight models with different datasets and different hyper parameters.

To combine the models we used `make_blend.py` script which optimizes directly MRR using validation predictions.

settings | mrr | coef
--- | --- | ----
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 5000, "drop_rate": 0.015, "feature_fraction": 0.7, "bagging_fraction": 0.8, "n_jobs": -2}} | 0.69068 | 0.51223
{"dataset_path_matrix": "data/eec9ef/Xcsr.h5", "dataset_path_meta": "data/eec9ef/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 5000, "drop_rate": 0.03, "feature_fraction": 0.7, "bagging_fraction": 0.8, "n_jobs": -2}} | 0.69026 | 0.76749
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.69018 | 0.37716
{"dataset_path_matrix": "data/04b695/Xcsr.h5", "dataset_path_meta": "data/04b695/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 5000, "drop_rate": 0.03, "feature_fraction": 0.7, "bagging_fraction": 0.8, "n_jobs": -2}} | 0.69011 | 0.47778
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 32, "min_child_samples": 5, "n_estimators": 4800, "n_jobs": -2}} | 0.69010 | 0.33917
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_sum_hessian_in_leaf": 0.0001, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.69006 | 0.25198
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "xgboost_dart_mode": true, "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.69002 | 0.58178
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 5000, "drop_rate": 0.015, "feature_fraction": 0.7, "bagging_fraction": 0.8, "n_jobs": -2}} | 0.69001 | 0.79173
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"max_position": 25, "boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.68995 | 0.00569
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"max_position": 25, "boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 3, "n_estimators": 3200, "n_jobs": -2}} | 0.68985 | 0.00161
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRankerMRR3", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.68970 | -0.12063
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 128, "min_child_samples": 5, "n_estimators": 1600, "n_jobs": -2}} | 0.68949 | 0.15376
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRankerMRR3", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 128, "min_child_samples": 5, "n_estimators": 1600, "n_jobs": -2}} | 0.68912 | -0.30801
{"dataset_path_matrix": "data/eec9ef/Xcsr.h5", "dataset_path_meta": "data/eec9ef/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 128, "min_child_samples": 5, "n_estimators": 1600, "n_jobs": -2}} | 0.68894 | 0.13886
{"dataset_path_matrix": "data/eec9ef/Xcsr.h5", "dataset_path_meta": "data/eec9ef/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 256, "min_child_samples": 5, "n_estimators": 1000, "drop_rate": 0.03, "feature_fraction": 0.5, "n_jobs": -2}} | 0.68860 | -0.32652
{"dataset_path_matrix": "data/f6e3ae/Xcsr.h5", "dataset_path_meta": "data/f6e3ae/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 128, "min_child_samples": 5, "n_estimators": 1600, "n_jobs": -2}} | 0.68825 | -0.10392
{"dataset_path_matrix": "data/04b695/Xcsr.h5", "dataset_path_meta": "data/04b695/meta.h5", "model_class": "LGBMRanker", "model_params": {"learning_rate": 0.1, "num_leaves": 62, "min_child_samples": 5, "n_estimators": 5000, "n_jobs": -2}} | 0.68783 | 0.17662
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"max_position": 25, "boosting_type": "dart", "learning_rate": 0.4, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.68782 | 0.40968
{"dataset_path_matrix": "data/9becae/Xcsr.h5", "dataset_path_meta": "data/9becae/meta.h5", "model_class": "LGBMRanker", "model_params": {“learning_rate”: 0.2, "num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68773 | 0.32795
{"dataset_path_matrix": "data/2744a6/Xcsr.h5", "dataset_path_meta": "data/2744a6/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68768 | -0.27577
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"learning_rate": 0.1, "num_leaves": 62, "min_child_samples": 5, "n_estimators": 5000, "n_jobs": -2}} | 0.68766 | -0.10544
{"dataset_path_matrix": "data/9becae/Xcsr.h5", "dataset_path_meta": "data/9becae/meta.h5", "model_class": "LGBMRanker3", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68755 | -0.46916
{"dataset_path_matrix": "data/04b695/Xcsr.h5", "dataset_path_meta": "data/04b695/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68743 | -0.29639
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"learning_rate": 0.1, "num_leaves": 128, "min_child_samples": 5, "n_estimators": 5000, "n_jobs": -2}} | 0.68743 | 0.37695
{"dataset_path_matrix": "data/5f87cf/Xcsr.h5", "dataset_path_meta": "data/5f87cf/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68735 | -0.04582
{"dataset_path_matrix": "data/9d2f56/Xcsr.h5", "dataset_path_meta": "data/9d2f56/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68731 | -0.16899
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker2", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68706 | 0.06479
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"max_position": 25, "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 3200, "n_jobs": -2}} | 0.68682 | 0.25987
{"dataset_path_matrix": "data/bdeaf8/Xcsr.h5", "dataset_path_meta": "data/bdeaf8/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68668 | -0.23845
{"dataset_path_matrix": "data/7ec17c/Xcsr.h5", "dataset_path_meta": "data/7ec17c/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68657 | -0.30018
{"dataset_path_matrix": "data/05bf57/Xcsr.h5", "dataset_path_meta": "data/05bf57/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68591 | 0.00519
{"dataset_path_matrix": "data/7b7222/Xcsr.h5", "dataset_path_meta": "data/7b7222/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68591 | 0.01476
{"dataset_path_matrix": "data/857d15/Xcsr.h5", "dataset_path_meta": "data/857d15/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68573 | 0.01326
{"dataset_path_matrix": "data/b1ca28/Xcsr.h5", "dataset_path_meta": "data/b1ca28/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68565 | -0.34195
{"dataset_path_matrix": "data/ed0938/Xcsr.h5", "dataset_path_meta": "data/ed0938/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68562 | 0.00619
{"dataset_path_matrix": "data/eb6c4a/Xcsr.h5", "dataset_path_meta": "data/eb6c4a/meta.h5", "model_class": "LGBMRanker", "model_params": {"num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68475 | 0.02687
{"dataset_path_matrix": "data/9becae/Xcsr.h5", "dataset_path_meta": "data/9becae/meta.h5", "model_class": "LGBMRanker", "model_params": {“learning_rate”: 0.4, "num_leaves": 62, "n_estimators": 1600, "n_jobs": -2}} | 0.68421 | 0.25348
