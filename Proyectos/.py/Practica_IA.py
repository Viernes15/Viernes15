# Importaciones básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from skopt import BayesSearchCV

# 1. Preparación de Datos de Ejemplo
def crear_dataset_ejemplo():
    # Dataset sintético
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convertir a DataFrame para simular datos reales
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Agregar algunas características categóricas
    df['categoria_1'] = np.random.choice(['A', 'B', 'C'], size=1000)
    df['categoria_2'] = np.random.choice(['X', 'Y'], size=1000)
    
    # Codificar variables categóricas
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df['categoria_1'] = le1.fit_transform(df['categoria_1'])
    df['categoria_2'] = le2.fit_transform(df['categoria_2'])
    
    return df, y

# Cargar datos
X, y = crear_dataset_ejemplo()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Dimensiones - Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

# 2. Implementación de XGBoost
def xgboost_basico(X_train, X_test, y_train, y_test):
    # Crear y entrenar modelo básico
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    # Predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost - Precisión básica: {accuracy:.4f}")
    return model, y_pred

def xgboost_optimizado(X_train, X_test, y_train, y_test):
    # Definir parámetros para búsqueda
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    # Crear modelo
    xgb_model = xgb.XGBClassifier(random_state=42)
    # Grid Search con validación cruzada
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    # Ejecutar búsqueda
    grid_search.fit(X_train, y_train)
    # Mejor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"XGBoost - Precisión optimizada: {accuracy:.4f}")
    return best_model, y_pred

# Ejecutar XGBoost
print("=== XGBOOST ===")
model_xgb_basic, pred_xgb_basic = xgboost_basico(X_train, X_test, y_train, y_test)
model_xgb_opt, pred_xgb_opt = xgboost_optimizado(X_train, X_test, y_train, y_test)

# 3. Implementación de LightGBM
def lightgbm_basico(X_train, X_test, y_train, y_test):
    # Crear y entrenar modelo básico
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    # Predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LightGBM - Precisión básica: {accuracy:.4f}")
    return model, y_pred

def lightgbm_optimizado(X_train, X_test, y_train, y_test):
    # Definir distribución de parámetros
    param_dist = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0]
    }
    # Crear modelo
    lgb_model = lgb.LGBMClassifier(random_state=42)
    # Randomized Search
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
    # Ejecutar búsqueda
    random_search.fit(X_train, y_train)
    # Mejor modelo
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"LightGBM - Precisión optimizada: {accuracy:.4f}")
    return best_model, y_pred

# Ejecutar LightGBM
print("\n=== LIGHTGBM ===")
model_lgb_basic, pred_lgb_basic = lightgbm_basico(X_train, X_test, y_train, y_test)
model_lgb_opt, pred_lgb_opt = lightgbm_optimizado(X_train, X_test, y_train, y_test)

# 4. Implementación de CatBoost
def catboost_basico(X_train, X_test, y_train, y_test):
    # Identificar columnas categóricas
    categorical_features = ['categoria_1', 'categoria_2']
    # Crear y entrenar modelo básico
    model = CatBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        cat_features=categorical_features,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    # Predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CatBoost - Precisión básica: {accuracy:.4f}")
    return model, y_pred

def catboost_optimizado(X_train, X_test, y_train, y_test):
    # Identificar columnas categóricas
    categorical_features = ['categoria_1', 'categoria_2']
    # Definir espacio de búsqueda
    search_spaces = {
        'depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'iterations': [100, 200, 300]
    }
    # Crear modelo
    cb_model = CatBoostClassifier(
        cat_features=categorical_features,
        random_state=42,
        verbose=0
    )
    # Bayesian Optimization
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
    # Ejecutar búsqueda
    bayes_search.fit(X_train, y_train)
    # Mejor modelo
    best_model = bayes_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejores parámetros: {bayes_search.best_params_}")
    print(f"CatBoost - Precisión optimizada: {accuracy:.4f}")
    return best_model, y_pred

# Ejecutar CatBoost
print("\n=== CATBOOST ===")
model_cb_basic, pred_cb_basic = catboost_basico(X_train, X_test, y_train, y_test)
model_cb_opt, pred_cb_opt = catboost_optimizado(X_train, X_test, y_train, y_test)

# 5. Validación Cruzada y Evaluación
def evaluacion_completa(modelos, nombres, X_test, y_test):
    """Evaluación comparativa de todos los modelos"""
    resultados = []
    for nombre, modelo in zip(nombres, modelos):
        # Predicciones
        y_pred = modelo.predict(X_test)
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        # Validación cruzada
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
        resultados.append({
            'Modelo': nombre,
            'Accuracy': accuracy,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
        print(f"\n{nombre}:")
        print(f" Accuracy: {accuracy:.4f}")
        print(f" CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return pd.DataFrame(resultados)

# Evaluar todos los modelos
modelos = [model_xgb_opt, model_lgb_opt, model_cb_opt]
nombres = ['XGBoost', 'LightGBM', 'CatBoost']
resultados_df = evaluacion_completa(modelos, nombres, X_test, y_test)
print("\n" + "="*50)
print("COMPARACIÓN FINAL DE MODELOS")
print("="*50)
print(resultados_df.to_string(index=False))

# 6. Selección de Características
def seleccion_caracteristas(modelo, X_train, y_train, X_test, nombre_modelo):
    """Selección recursiva de características"""
    # RFE con validación cruzada
    selector = RFECV(
        estimator=modelo,
        step=1,
        cv=5,
        scoring='accuracy',
        min_features_to_select=10,
        n_jobs=-1
    )
    # Ajustar selector
    selector = selector.fit(X_train, y_train)
    # Transformar datos
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    print(f"\n{nombre_modelo} - Selección de Características:")
    print(f"Características seleccionadas: {selector.n_features_}")
    print(f"Características más importantes: {np.where(selector.support_)[0]}")
    return X_train_selected, X_test_selected, selector

# Aplicar selección de características a cada modelo
for modelo, nombre in zip(modelos, nombres):
    X_train_sel, X_test_sel, selector = seleccion_caracteristas(
        modelo, X_train, y_train, X_test, nombre
    )
    # Reentrenar modelo con características seleccionadas
    modelo_mejorado = modelo.fit(X_train_sel, y_train)
    y_pred_mejorado = modelo_mejorado.predict(X_test_sel)
    accuracy_mejorado = accuracy_score(y_test, y_pred_mejorado)
    print(f"Accuracy después de selección: {accuracy_mejorado:.4f}\n")

# 7. Visualización de Resultados
def visualizar_resultados(resultados_df):
    """Visualización comparativa de los resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Gráfico 1: Comparación de Accuracy
    axes[0, 0].bar(resultados_df['Modelo'], resultados_df['Accuracy'])
    axes[0, 0].set_title('Comparación de Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    # Gráfico 2: Comparación de Validación Cruzada
    axes[0, 1].bar(resultados_df['Modelo'], resultados_df['CV Mean'],
                   yerr=resultados_df['CV Std'], capsize=5)
    axes[0, 1].set_title('Validación Cruzada (5-fold)')
    axes[0, 1].set_ylabel('CV Score')
    # Gráfico 3: Importancia de características (XGBoost)
    importancias = model_xgb_opt.feature_importances_
    indices = np.argsort(importancias)[::-1]
    features = X_train.columns
    axes[1, 0].bar(range(10), importancias[indices][:10])
    axes[1, 0].set_title('Top 10 Características Importantes (XGBoost)')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_xticklabels(features[indices][:10], rotation=45)
    # Gráfico 4: Matriz de confusión del mejor modelo
    mejor_modelo_idx = resultados_df['Accuracy'].idxmax()
    mejor_modelo_nombre = resultados_df.loc[mejor_modelo_idx, 'Modelo']
    mejor_modelo = modelos[mejor_modelo_idx]
    y_pred_best = mejor_modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1])
    axes[1, 1].set_title(f'Matriz de Confusión - {mejor_modelo_nombre}')
    axes[1, 1].set_xlabel('Predicho')
    axes[1, 1].set_ylabel('Real')
    plt.tight_layout()
    plt.show()

# Generar visualizaciones
visualizar_resultados(resultados_df)