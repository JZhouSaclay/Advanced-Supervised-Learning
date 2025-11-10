# Advanced-Supervised-Learning

1. - # Hotel Booking Cancellation Prediction - Classification Task

   ## 1. Project Overview

   This is a multi-class classification machine learning task, aiming to predict the final status of a hotel booking based on its detailed information:

   - **Class 0: Check-out**
   - **Class 1: Cancel**
   - **Class 2: No-Show**

   This project is the **Classification** task portion of the course data challenge.

   The core modeling strategy is **Stacking**, using a customized L0 layer (LGBM, CatBoost, RandomForest, Keras NN) and an L1 meta-model (LGBM-Meta or KerasNN-Meta) to make the final prediction.

   ## 2. Repository Contents

   - `Booking_code.ipynb`: The main Jupyter Notebook containing all the code. It covers data loading, advanced feature engineering, 10-fold cross-validation (K-Fold) training for the 4 L0 models, training for the 2 L1 meta-models, OOF (Out-of-Fold) evaluation, and the generation of all visualization charts (confusion matrices, feature importance, learning curves).
   - `README.md`: This guide file.

   ## 3. Execution Guide

   ### 3.1. Prerequisites

   This project depends on several Python libraries. It is **strongly recommended** to run this notebook in a **GPU runtime**(like Google Colab), as the CatBoost and Keras NN models in L0, as well as the KerasNN-Meta model in L1, are configured for GPU acceleration.

   You can install all required libraries via `pip`:

   ```
   pip install pandas numpy scikit-learn lightgbm xgboost catboost tensorflow tqdm matplotlib seaborn jupyter
   ```

   ### 3.2. Data

   **Note:** As per the requirements, the data files (`train_data.csv`, `test_data.csv`) are **not** included in this repository.

   1. **Get the Data:** Please download `train_data.csv` and `test_data.csv` from your course platform.

   2. **Place the Data:** The simplest method is to place the `train_data.csv` and `test_data.csv` files in the **same root directory** as `Booking_code.ipynb`. The code will find them automatically.

   3. **Modify Paths:** If your data is stored in a different location, please open `Booking_code.ipynb` and modify the following two lines in **Cell 4** to point to the correct local path:

      ```
      # Cell 4
      train_path = "your/path/to/train_data.csv"
      test_path  = "your/path/to/test_data.csv"
      ```

   ### 3.3. Running the Code

   1. After setting up the environment and data, open `Booking_code.ipynb` in a Jupyter environment (e.g., Google Colab, VS Code, Jupyter Lab).
   2. **Run all cells in order from top to bottom.**
   3. The notebook will execute the complete Stacking pipeline. This will take some time, as the L0 layer trains 3 or 4 models, each with 10 folds (40 models in total).
   4. **Final Outputs:**
      - **Submission Files:** Upon completion, two final submission files will be generated in the root directory: `submission_lgbmcbnn.csv` (from the L1 LGBM-Meta) and `last_soumission.csv` (from the L1 KerasNN-Meta).
      - **Visualizations:** The optional cells at the end of the notebook will generate and save OOF confusion matrices, L0 feature importance plots, and learning curves as `.png` files.

