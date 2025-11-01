import numpy as np                     
import pandas as pd                   
import matplotlib.pyplot as plt         
import seaborn as sns                   


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from skopt import BayesSearchCV




def crear_dataset_ejemplo():
    """Genera un dataset sintético con variables numéricas y categóricas"""
    
    # Crear un dataset sintético de clasificación
    X, y = make_classification(
        n_samples=1000,         
        n_features=20,          
        n_informative=15,       
        n_redundant=5,          
        n_clusters_per_class=1,
        random_state=42
    )

    # Crear nombres de las columnas
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)

    # Agregar dos columnas categóricas simuladas
    df['categoria_1'] = np.random.choice(['A', 'B', 'C'], size=1000)
    df['categoria_2'] = np.random.choice(['X', 'Y'], size=1000)

    # Codificar variables categóricas con LabelEncoder
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df['categoria_1'] = le1.fit_transform(df['categoria_1'])
    df['categoria_2'] = le2.fit_transform(df['categoria_2'])

    return df, y

# Crear dataset
X, y = crear_dataset_ejemplo()

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dimensiones - Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")



def xgboost_basico(X_train, X_test, y_train, y_test):
    """Entrena un modelo básico de XGBoost"""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost - Precisión básica: {accuracy:.4f}")
    return model, y_pred


def xgboost_optimizado(X_train, X_test, y_train, y_test):
    """Optimiza XGBoost usando GridSearchCV"""
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }

    xgb_model = xgb.XGBClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"XGBoost - Precisión optimizada: {accuracy:.4f}")

    return best_model, y_pred


print("\n=== XGBOOST ===")
model_xgb_basic, _ = xgboost_basico(X_train, X_test, y_train, y_test)
model_xgb_opt, _ = xgboost_optimizado(X_train, X_test, y_train, y_test)



def lightgbm_basico(X_train, X_test, y_train, y_test):
    """Entrena un modelo básico de LightGBM"""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LightGBM - Precisión básica: {accuracy:.4f}")
    return model, y_pred


def lightgbm_optimizado(X_train, X_test, y_train, y_test):
    """Optimiza LightGBM usando RandomizedSearchCV"""
    param_dist = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0]
    }

    lgb_model = lgb.LGBMClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"LightGBM - Precisión optimizada: {accuracy:.4f}")

    return best_model, y_pred


print("\n=== LIGHTGBM ===")
model_lgb_basic, _ = lightgbm_basico(X_train, X_test, y_train, y_test)
model_lgb_opt, _ = lightgbm_optimizado(X_train, X_test, y_train, y_test)




def catboost_basico(X_train, X_test, y_train, y_test):
    """Entrena un modelo básico de CatBoost"""
    categorical_features = ['categoria_1', 'categoria_2']
    model = CatBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        cat_features=categorical_features,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CatBoost - Precisión básica: {accuracy:.4f}")
    return model, y_pred


def catboost_optimizado(X_train, X_test, y_train, y_test):
    """Optimiza CatBoost usando búsqueda bayesiana"""
    categorical_features = ['categoria_1', 'categoria_2']

    search_spaces = {
        'depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'iterations': [100, 200, 300]
    }

    cb_model = CatBoostClassifier(cat_features=categorical_features, random_state=42, verbose=0)

    bayes_search = BayesSearchCV(
        estimator=cb_model,
        search_spaces=search_spaces,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    bayes_search.fit(X_train, y_train)
    best_model = bayes_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Mejores parámetros: {bayes_search.best_params_}")
    print(f"CatBoost - Precisión optimizada: {accuracy:.4f}")

    return best_model, y_pred


print("\n=== CATBOOST ===")
model_cb_basic, _ = catboost_basico(X_train, X_test, y_train, y_test)
model_cb_opt, _ = catboost_optimizado(X_train, X_test, y_train, y_test)




def evaluacion_completa(modelos, nombres, X_test, y_test):
    """Evalúa todos los modelos comparando Accuracy y Validación Cruzada"""
    resultados = []
    for nombre, modelo in zip(nombres, modelos):
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
        resultados.append({
            'Modelo': nombre,
            'Accuracy': accuracy,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
        print(f"\n{nombre}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return pd.DataFrame(resultados)


modelos = [model_xgb_opt, model_lgb_opt, model_cb_opt]
nombres = ['XGBoost', 'LightGBM', 'CatBoost']

resultados_df = evaluacion_completa(modelos, nombres, X_test, y_test)

print("\n" + "="*50)
print("COMPARACIÓN FINAL DE MODELOS")
print("="*50)
print(resultados_df.to_string(index=False))



def visualizar_resultados(resultados_df):
    """Genera gráficos comparativos entre los modelos"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfico 1: Accuracy
    axes[0, 0].bar(resultados_df['Modelo'], resultados_df['Accuracy'])
    axes[0, 0].set_title('Comparación de Accuracy')
    axes[0, 0].set_ylabel('Accuracy')

    # Gráfico 2: Validación cruzada
    axes[0, 1].bar(resultados_df['Modelo'], resultados_df['CV Mean'],
                   yerr=resultados_df['CV Std'], capsize=5)
    axes[0, 1].set_title('Validación Cruzada (5-fold)')
    axes[0, 1].set_ylabel('CV Score')

    # Gráfico 3: Importancia de características (XGBoost)
    importancias = model_xgb_opt.feature_importances_
    indices = np.argsort(importancias)[::-1][:10]
    features = X_train.columns
    axes[1, 0].bar(range(10), importancias[indices][:10])
    axes[1, 0].set_title('Top 10 Características (XGBoost)')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_xticklabels(features[indices][:10], rotation=45)

    # Gráfico 4: Matriz de confusión del mejor modelo
    mejor_modelo_idx = resultados_df['Accuracy'].idxmax()
    mejor_modelo = modelos[mejor_modelo_idx]
    mejor_nombre = resultados_df.loc[mejor_modelo_idx, 'Modelo']
    y_pred_best = mejor_modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1])
    axes[1, 1].set_title(f'Matriz de Confusión - {mejor_nombre}')
    axes[1, 1].set_xlabel('Predicho')
    axes[1, 1].set_ylabel('Real')

    plt.tight_layout()
    plt.show()


# Ejecutar visualización
visualizar_resultados(resultados_df)
