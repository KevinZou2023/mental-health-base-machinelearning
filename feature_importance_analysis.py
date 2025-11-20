# -*- coding: utf-8 -*-
# 1. 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# 检查并导入 GPU 加速库
try:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    print("CatBoost, LightGBM, XGBoost 库已成功导入。")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 catboost, lightgbm, xgboost 库。")
    exit()


def check_gpu_availability():
    """
    检查各个库的GPU可用性
    """
    print("=" * 60)
    print("GPU 可用性检查:")
    print("=" * 60)
    
    # 检查 XGBoost GPU
    try:
        import xgboost as xgb
        print(f"✓ XGBoost 版本: {xgb.__version__}")
        use_cuda = xgb.build_info().get('USE_CUDA', False)
        print(f"  CUDA 支持: {'是' if use_cuda else '否'}")
        if use_cuda:
            cuda_version = xgb.build_info().get('CUDA_VERSION', [])
            if cuda_version:
                print(f"  内置 CUDA 版本: {'.'.join(map(str, cuda_version))}")
    except Exception as e:
        print(f"✗ XGBoost GPU 检查失败: {e}")
    
    # 检查 LightGBM GPU
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM 版本: {lgb.__version__}")
        # 尝试创建一个简单的GPU数据集来测试
        try:
            test_data = lgb.Dataset(np.random.rand(10, 5), label=np.random.randint(0, 2, 10))
            print(f"  GPU 支持: 需要在训练时验证")
        except:
            print(f"  GPU 支持: 未知")
    except Exception as e:
        print(f"✗ LightGBM GPU 检查失败: {e}")
    
    # 检查 CatBoost GPU
    try:
        from catboost import CatBoostClassifier
        import catboost
        print(f"✓ CatBoost 版本: {catboost.__version__}")
        print(f"  GPU 支持: 是 (task_type='GPU')")
    except Exception as e:
        print(f"✗ CatBoost GPU 检查失败: {e}")
    
    # 检查 CUDA
    try:
        import torch
        print(f"✓ PyTorch CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
    except ImportError:
        print("  PyTorch 未安装（可选）")
    
    print("=" * 60)
    print()


def main():
    """
    主函数，执行完整的特征重要性分析流程。
    """
    # 设置环境变量以优化CPU并行性能
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # OpenMP线程数
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())  # Intel MKL线程数
    os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())  # OpenBLAS线程数
    
    print(f"检测到 {os.cpu_count()} 个CPU核心，已启用多线程并行优化")
    
    warnings.filterwarnings('ignore')
    sns.set(style="whitegrid", font_scale=1.1)
    
    # 检查 GPU 可用性
    check_gpu_availability()

    # 设置中文字体，以防图表中文乱码
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体 'SimHei' 设置成功。")
    except Exception:
        print("警告: 未找到 'SimHei' 字体，图表中的中文可能显示为方框。")
        print("您可以尝试安装 'SimHei' 字体，或在代码中更换为其他已安装的中文字体。")

    # 2. 定义配置和模型超参数
    class CFG:
        train_path = 'dataset/train.csv'
        original_data_path = 'dataset/final_depression_dataset_1.csv'
        target = 'Depression'
        seed = 42

    # 从主 Notebook 复制过来的超参数
    # 启用多线程以充分利用CPU资源
    lr_params = {
        "C": 5.559, "max_iter": 1000, "n_jobs": -1, "penalty": "l2", "random_state": CFG.seed, "solver": "lbfgs"  # lbfgs支持多线程
    }

    cb_params = {
        'iterations': 2372, 'learning_rate': 0.0514, 'depth': 4, 'l2_leaf_reg': 4.44, 'min_child_samples': 146,
        'random_state': CFG.seed, 'verbose': False, 'task_type': "GPU", 'devices': "0"
    }

    # XGBoost 参数 - 启用 GPU 加速
    xgb_params = {
        'n_estimators': 3853, 'colsample_bytree': 0.18, 'gamma': 3.63, 'max_depth': 16,
        'min_child_weight': 34, 'reg_alpha': 7.99, 'reg_lambda': 46.83, 'subsample': 0.91,
        'random_state': CFG.seed, 'verbosity': 0, 'device': 'cuda', 'tree_method': 'hist'
    }

    gpu_dict = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}

    lgbm_params = {
        'boosting_type': 'gbdt', 'n_estimators': 244, 'learning_rate': 0.099, 'num_leaves': 122,
        'colsample_bytree': 0.18, 'min_child_samples': 105, 'reg_alpha': 8.66, 'reg_lambda': 3.57,
        'random_state': CFG.seed, 'verbose': -1, **gpu_dict
    }

    lgbm_goss_params = {
        'boosting_type': 'goss', 'n_estimators': 966, 'learning_rate': 0.046, 'num_leaves': 159,
        'colsample_bytree': 0.119, 'min_child_samples': 283, 'reg_alpha': 8.16, 'reg_lambda': 9.86,
        'random_state': CFG.seed, 'verbose': -1, **gpu_dict
    }

    lgbm_dart_params = {
        'boosting_type': 'dart', 'n_estimators': 1454, 'learning_rate': 0.091, 'num_leaves': 163,
        'colsample_bytree': 0.118, 'min_child_samples': 225, 'reg_alpha': 2.96, 'reg_lambda': 9.78,
        'random_state': CFG.seed, 'verbose': -1, **gpu_dict
    }

    # GradientBoosting 和 AdaBoost 原始参数 - 启用CPU多线程并行
    # 注意：scikit-learn的GradientBoosting不支持n_jobs参数，但可以优化其他方面
    gb_params = {
        'n_estimators': 1048, 'learning_rate': 0.129, 'max_depth': 82, 'max_features': 0.98,
        'min_samples_leaf': 0.0015, 'min_samples_split': 0.338, 'subsample': 0.955, 'random_state': CFG.seed
    }
    
    adb_params = {
        'n_estimators': 410, 'learning_rate': 1.517, 'random_state': CFG.seed, 'algorithm': 'SAMME'
    }

    # 3. 数据准备函数
    def get_data(model_type):
        train = pd.read_csv(CFG.train_path)
        original = pd.read_csv(CFG.original_data_path)
        original[CFG.target] = original[CFG.target].map({'Yes': 1, 'No': 0})
        
        y_train = train[CFG.target]
        X_train = train.drop([CFG.target, 'id', 'Name'], axis=1)
        y_original = original[CFG.target]
        X_original = original.drop([CFG.target, 'Name'], axis=1)[X_train.columns]
        
        X_aug = pd.concat([X_train, X_original], ignore_index=True)
        y_aug = pd.concat([y_train, y_original], ignore_index=True)

        cat_cols = X_aug.select_dtypes(include='object').columns.tolist()
        
        if model_type == 'cb':
            X_aug[cat_cols] = X_aug[cat_cols].astype(str).fillna('missing')
            return X_aug, y_aug, X_aug.columns.tolist()

        if 'lgbm' in model_type:
            for col in cat_cols:
                X_aug[col] = X_aug[col].astype('category')
            return X_aug, y_aug, X_aug.columns.tolist()

        if model_type in ['xgb', 'gb', 'adb']:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_aug[cat_cols] = encoder.fit_transform(X_aug[cat_cols])
            
            imputer = SimpleImputer(strategy='mean')
            X_aug = pd.DataFrame(imputer.fit_transform(X_aug), columns=X_aug.columns)
            return X_aug, y_aug, X_aug.columns.tolist()

        if model_type == 'lr':
            num_cols = X_aug.select_dtypes(include=np.number).columns.tolist()
            imputer_num = SimpleImputer(strategy='mean')
            X_aug[num_cols] = imputer_num.fit_transform(X_aug[num_cols])

            imputer_cat = SimpleImputer(strategy='most_frequent')
            X_aug[cat_cols] = imputer_cat.fit_transform(X_aug[cat_cols])
            onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_onehot = onehot_encoder.fit_transform(X_aug[cat_cols])
            onehot_feature_names = onehot_encoder.get_feature_names_out(cat_cols)
            X_onehot_df = pd.DataFrame(X_onehot, columns=onehot_feature_names, index=X_aug.index)

            X_processed = pd.concat([X_aug.drop(cat_cols, axis=1), X_onehot_df], axis=1)
            return X_processed, y_aug, X_processed.columns.tolist()

        return X_aug, y_aug, X_aug.columns.tolist()

    # 4. 训练模型并提取特征重要性
    models_to_analyze = {
        'LogisticRegression': (LogisticRegression(**lr_params), 'lr'),
        'CatBoost': (CatBoostClassifier(**cb_params), 'cb'),
        'XGBoost': (XGBClassifier(**xgb_params), 'xgb'),
        'LightGBM_gbdt': (LGBMClassifier(**lgbm_params), 'lgbm_gbdt'),
        'LightGBM_goss': (LGBMClassifier(**lgbm_goss_params), 'lgbm_goss'),
        'LightGBM_dart': (LGBMClassifier(**lgbm_dart_params), 'lgbm_dart'),
        'GradientBoosting': (GradientBoostingClassifier(**gb_params), 'gb'),
        'AdaBoost': (AdaBoostClassifier(**adb_params), 'adb')
    }

    all_feature_importances = {}
    
    # 创建保存图像的文件夹
    output_dir = 'feature_importance_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建文件夹: {output_dir}")

    import time
    
    for model_name, (model, model_type) in models_to_analyze.items():
        print(f'--- 正在训练: {model_name} ---')
        start_time = time.time()
        
        try:
            X, y, feature_names = get_data(model_type)
            # 特别处理 CatBoost 的分类特征
            if model_name == 'CatBoost':
                # CatBoost 需要知道哪些列是分类特征
                cat_feature_indices = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]
                model.fit(X, y, cat_features=cat_feature_indices)
            else:
                model.fit(X, y)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                print(f'无法为 {model_name} 提取特征重要性。')
                continue
                
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            all_feature_importances[model_name] = importance_df
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title(f'{model_name} - 特征重要性 (Top 20)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_importance.png'))
            plt.close() # 关闭图像，防止占用内存
            elapsed_time = time.time() - start_time
            print(f'{model_name} 的图表已保存。耗时: {elapsed_time:.2f}秒')

        except Exception as e:
            print(f"训练或分析 {model_name} 时发生错误: {e}")
            # 如果是GPU相关的错误，给出提示
            if 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                print(f"错误可能与 GPU 加速有关。请检查您的 CUDA 环境和 {model_name} 的 GPU 版本是否正确安装。")
            continue

    # 5. 综合特征重要性分析
    print('\n--- 开始综合特征重要性分析 ---')

    lr_importances = all_feature_importances.get('LogisticRegression')
    if lr_importances is not None:
        original_cat_cols = pd.read_csv(CFG.train_path).select_dtypes(include='object').columns.tolist()
        original_num_cols = pd.read_csv(CFG.train_path).select_dtypes(include=np.number).columns.drop(['id', 'Depression']).tolist()
        
        aggregated_lr_importance = pd.DataFrame(index=original_num_cols + original_cat_cols, columns=['Importance'], data=0.0)

        for _, row in lr_importances.iterrows():
            feature_name = row['Feature']
            importance = row['Importance']
            is_onehot_feature = False
            for cat_col in original_cat_cols:
                if feature_name.startswith(cat_col + '_'):
                    aggregated_lr_importance.loc[cat_col, 'Importance'] += importance
                    is_onehot_feature = True
                    break
            if not is_onehot_feature:
                 if feature_name in aggregated_lr_importance.index:
                    aggregated_lr_importance.loc[feature_name, 'Importance'] = importance
        
        aggregated_lr_importance.reset_index(inplace=True)
        aggregated_lr_importance.rename(columns={'index': 'Feature'}, inplace=True)
        all_feature_importances['LogisticRegression_Aggregated'] = aggregated_lr_importance

    final_importance_df = None

    for model_name, df in all_feature_importances.items():
        if model_name == 'LogisticRegression': continue
        
        df = df.rename(columns={'Importance': model_name})
        
        scaler = MinMaxScaler()
        df[model_name] = scaler.fit_transform(df[[model_name]])
        
        if final_importance_df is None:
            final_importance_df = df
        else:
            final_importance_df = pd.merge(final_importance_df, df, on='Feature', how='outer')

    if final_importance_df is not None:
        final_importance_df = final_importance_df.fillna(0)
        
        model_cols = [col for col in final_importance_df.columns if col != 'Feature']
        final_importance_df['Mean_Importance'] = final_importance_df[model_cols].mean(axis=1)
        
        sorted_final_importance = final_importance_df.sort_values(by='Mean_Importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Mean_Importance', y='Feature', data=sorted_final_importance.head(20))
        plt.title('综合特征重要性 (所有模型归一化后平均)', fontsize=16)
        plt.xlabel('归一化后的平均重要性分数', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ZZ_Overall_Mean_Importance.png'))
        plt.close()
        
        print('\n分析完成！所有独立模型的特征重要性图及最终的综合图已保存至 `feature_importance_plots` 文件夹中。')
    else:
        print("未能生成任何特征重要性数据，无法进行综合分析。")

if __name__ == '__main__':
    main()