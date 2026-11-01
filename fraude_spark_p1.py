"""
Práctica 1 — Detección de fraude con Spark MLlib.

Requisitos: PySpark, pandas, numpy y matplotlib.
Por defecto busca cleaned.csv en el directorio de ejecución.
Se puede configurar DATA_PATH y OUTPUT_DIR mediante variables de entorno.
"""


import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.functions import vector_to_array

SEED = 42
DATA_PATH = os.environ.get("DATA_PATH", "cleaned.csv")
OUT = Path(os.environ.get("OUTPUT_DIR", "resultados"))
OUT.mkdir(parents=True, exist_ok=True)

COMPARAR_SALDOS_POSTERIORES = True
GUARDAR_MODELO = True

spark = (
    SparkSession.builder
    .appName("AAGE_Practica1_Fraude")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Versión de Spark:", spark.version)
print("Directorio de resultados:", OUT.resolve())


if not Path(DATA_PATH).is_file() and not DATA_PATH.startswith(("hdfs://", "s3a://", "dbfs:/")):
    raise FileNotFoundError(
        f"No encuentro {DATA_PATH!r}. Coloca cleaned.csv junto al notebook "
        "o configura la variable de entorno DATA_PATH."
    )

df = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
N = df.count()

LABEL_CANDIDATES = ["isFraud", "is_fraud", "fraud", "label", "Class", "class"]
TARGET = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
if TARGET is None:
    raise ValueError(f"No se encuentra la columna objetivo. Disponibles: {df.columns}")

df = df.withColumn("_label", F.col(TARGET).cast("double"))
if df.filter(F.col("_label").isNull() | ~F.col("_label").isin(0.0, 1.0)).limit(1).count():
    raise ValueError("La etiqueta de fraude debe contener únicamente los valores 0 y 1.")

print(f"Filas: {N:,} | Columnas originales: {len(df.columns)-1} | Objetivo: {TARGET}")
df.printSchema()
df.show(5, truncate=False)


numeric_cols = [
    c for c in df.columns
    if c not in (TARGET, "_label") and isinstance(df.schema[c].dataType, NumericType)
]
cat_cols = [c for c in df.columns if c not in numeric_cols + [TARGET, "_label"]]

print("Variables numéricas:", numeric_cols)
print("Variables de texto:", cat_cols)

missing_exprs = []
for colname in df.columns:
    if colname == "_label":
        continue
    missing = F.col(colname).isNull()
    if colname in numeric_cols:
        missing = missing | F.isnan(F.col(colname).cast("double"))
    missing_exprs.append(
        F.sum(F.when(missing, 1).otherwise(0)).alias(colname)
    )
missing_row = df.agg(*missing_exprs).first().asDict()
quality = pd.DataFrame([
    {"variable": c, "tipo": dict(df.dtypes)[c], "nulos": int(missing_row[c] or 0),
     "porcentaje_nulos": round(100 * (missing_row[c] or 0) / N, 4)}
    for c in df.columns if c != "_label"
]).sort_values("porcentaje_nulos", ascending=False)
print(quality.to_string(index=False))
quality.to_csv(OUT / "calidad_datos.csv", index=False)


class_table = (
    df.groupBy("_label").count().orderBy("_label").toPandas()
)
if set(class_table["_label"]) != {0.0, 1.0}:
    raise ValueError("El conjunto de datos debe contener ambas clases (0 y 1).")
class_table["clase"] = class_table["_label"].map({0.0: "Normal", 1.0: "Fraude"})
prevalence = float(class_table.loc[class_table["_label"] == 1, "count"].iloc[0]) / N
print(class_table[["clase", "count"]].to_string(index=False))
print(f"Porcentaje de fraude: {prevalence:.4%}")
print(f"AUC-PR de un clasificador aleatorio (aprox.): {prevalence:.6f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(class_table["clase"], class_table["count"])
ax.set_yscale("log")
ax.set_title("Cantidad de transacciones por clase")
ax.set_ylabel("Número de operaciones (escala logarítmica)")
plt.tight_layout()
plt.savefig(OUT / "01_desbalance.png", dpi=140)
plt.show()


eda_cols = [c for c in ["amount", "oldbalanceOrg", "oldbalanceDest"] if c in numeric_cols]
if not eda_cols:
    raise ValueError("No se encontraron las variables financieras esperadas para el análisis.")

eda_rows = []
for c in eda_cols:
    stats = (
        df.groupBy("_label")
        .agg(
            F.mean(c).alias("media"),
            F.expr(f"percentile_approx(`{c}`, 0.5, 1000)").alias("mediana")
        )
        .collect()
    )
    for r in stats:
        eda_rows.append({"variable": c, "clase": "Fraude" if r["_label"] == 1 else "Normal",
                         "media": r["media"], "mediana": r["mediana"]})
eda_table = pd.DataFrame(eda_rows)
print(eda_table.round(2).to_string(index=False))
eda_table.to_csv(OUT / "eda_importes.csv", index=False)



if "type" in df.columns:
    types = (
        df.groupBy("type")
        .agg(F.count("*").alias("operaciones"), F.sum("_label").alias("fraudes"))
        .withColumn("tasa_fraude", F.col("fraudes") / F.col("operaciones"))
        .orderBy(F.desc("operaciones"))
        .toPandas()
    )
    print(types.to_string(index=False))
    types.to_csv(OUT / "fraude_por_tipo.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(types["type"].astype(str), types["tasa_fraude"] * 100)
    ax.set_title("Tasa de fraude según el tipo de operación")
    ax.set_ylabel("Fraudes / operaciones (%)")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(OUT / "02_fraude_por_tipo.png", dpi=140)
    plt.show()
else:
    print("El dataset no incluye 'type'. Se omite este análisis.")


finance_cols = [
    c for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    if c in numeric_cols
]
correlations = []
for i, a in enumerate(finance_cols):
    for b in finance_cols[i+1:]:
        r = df.stat.corr(a, b)
        correlations.append({"variable_1": a, "variable_2": b, "pearson": r})

corr_table = pd.DataFrame(correlations)
if not corr_table.empty:
    corr_table["abs_pearson"] = corr_table["pearson"].abs()
    corr_table = corr_table.sort_values("abs_pearson", ascending=False)
    print(corr_table.head(8).round(4).to_string(index=False))
    corr_table.to_csv(OUT / "correlaciones_financieras.csv", index=False)
else:
    print("No hay suficientes variables financieras para calcular correlaciones.")



safe_candidates = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "step", "day_of_sim", "hour_of_day", "is_merchant_dest"
]
NUM = [c for c in safe_candidates if c in numeric_cols]
CAT = [c for c in ["type"] if c in df.columns]

if all(c in NUM for c in ("step", "day_of_sim", "hour_of_day")):
    NUM.remove("step")

POST = [c for c in ["newbalanceOrig", "newbalanceDest"] if c in numeric_cols]
if not NUM:
    raise ValueError("No hay variables numéricas admitidas para entrenar.")

print("Numéricas del modelo operativo:", NUM)
print("Categóricas del modelo operativo:", CAT)
print("Saldos posteriores (SOLO experimento):", POST)



all_needed = list(dict.fromkeys(NUM + POST + CAT))

work = df.select("_label", *all_needed).withColumnRenamed(
    "_label", "label"
)

for c in NUM + POST:
    work = work.withColumn(c, F.col(c).cast("double"))
    work = work.withColumn(
        c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c))
    )

for c in CAT:
    work = work.withColumn(
        c,
        F.coalesce(F.col(c).cast("string"), F.lit("__MISSING__"))
    )

train, val, test = work.randomSplit(
    [0.70, 0.15, 0.15],
    seed=SEED
)

for name, subset in [
    ("Entrenamiento", train),
    ("Validación", val),
    ("Test", test)
]:
    clases = {
        int(r["label"]): int(r["count"])
        for r in subset.groupBy("label").count().collect()
    }

    if 0 not in clases or 1 not in clases:
        raise ValueError(f"Falta alguna clase en {name}")

    total = sum(clases.values())

    print(
        f"{name}: {total:,} operaciones | "
        f"Normales: {clases[0]:,} | "
        f"Fraudes: {clases[1]:,}"
    )



def make_preprocessing(numeric_features, categorical_features):
    """Crea el mismo preprocesamiento para cualquier algoritmo."""
    steps = []
    output_num = [f"{c}_imp" for c in numeric_features]
    steps.append(Imputer(inputCols=numeric_features, outputCols=output_num, strategy="median"))

    output_cat = []
    for c in categorical_features:
        steps.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))
        steps.append(OneHotEncoder(inputCols=[f"{c}_idx"],
                                   outputCols=[f"{c}_ohe"], handleInvalid="keep"))
        output_cat.append(f"{c}_ohe")

    steps.append(VectorAssembler(inputCols=output_num + output_cat,
                                 outputCol="features", handleInvalid="error"))
    return steps

pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                        metricName="areaUnderPR")
roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                         metricName="areaUnderROC")

def evaluate_predictions(pred):
    """Calcula las métricas de fraude y la matriz de confusión (sin pandas masivo)."""
    pairs = pred.groupBy("label", "prediction").count().collect()
    cm = {(int(r["label"]), int(r["prediction"])): int(r["count"]) for r in pairs}
    tn = cm.get((0, 0), 0)
    fp = cm.get((0, 1), 0)
    fn = cm.get((1, 0), 0)
    tp = cm.get((1, 1), 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auc_pr": float(pr_eval.evaluate(pred)),
        "auc_roc": float(roc_eval.evaluate(pred)),
        "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

print("Pipeline y evaluación preparados.")



lr_params = {"labelCol": "label", "featuresCol": "features",
             "maxIter": 50, "regParam": 0.05, "standardization": True}

algorithms = {
    "RegresionLogistica": LogisticRegression(**lr_params),
    "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features",
                                           numTrees=60, maxDepth=7, seed=SEED),
    "GBT": GBTClassifier(labelCol="label", featuresCol="features",
                         maxIter=35, maxDepth=5, seed=SEED),
}

trained_models = {}
model_results = []

for name, algorithm in algorithms.items():
    print("Entrenando:", name)
    t0 = time.perf_counter()
    pipeline = Pipeline(stages=make_preprocessing(NUM, CAT) + [algorithm])
    model = pipeline.fit(train)
    metrics = evaluate_predictions(model.transform(val))
    trained_models[name] = model
    model_results.append({"modelo": name, "segundos": round(time.perf_counter()-t0, 1), **metrics})
    print(f"  AUC-PR={metrics['auc_pr']:.4f} | Recall={metrics['recall']:.4f} | "
          f"Precision={metrics['precision']:.4f}")

print("\nSe han entrenado los tres modelos.")



rf_tuned = RandomForestClassifier(
    labelCol="label", featuresCol="features", seed=SEED
)
gbt_tuned = GBTClassifier(
    labelCol="label", featuresCol="features", seed=SEED
)

searches = {
    "RandomForest": (
        rf_tuned,
        [(rf_tuned.numTrees, [40, 60]),
         (rf_tuned.maxDepth, [5, 7])]
    ),
    "GBT": (
        gbt_tuned,
        [(gbt_tuned.maxIter, [20, 35]),
         (gbt_tuned.maxDepth, [3, 5])]
    ),
}

model_results = [r for r in model_results
                 if r["modelo"] not in ("RandomForest_optimizado", "GBT_optimizado")]

for name, (algorithm, parameters) in searches.items():
    grid_builder = ParamGridBuilder()
    for parameter, values in parameters:
        grid_builder.addGrid(parameter, values)
    param_grid = grid_builder.build()

    tuner = TrainValidationSplit(
        estimator=Pipeline(stages=make_preprocessing(NUM, CAT) + [algorithm]),
        estimatorParamMaps=param_grid,
        evaluator=pr_eval,
        trainRatio=0.80,
        seed=SEED,
        parallelism=1
    )

    print(f"\nOptimizando {name}: {len(param_grid)} combinaciones")
    t0 = time.perf_counter()
    tuned = tuner.fit(train)
    opt_seconds = round(time.perf_counter() - t0, 1)

    best_grid_idx = int(np.argmax(tuned.validationMetrics))
    print("Mejor combinación (validación interna):")
    for parameter, value in param_grid[best_grid_idx].items():
        print(" -", parameter.name, "=", value)
    print("AUC-PR validación interna:",
          round(float(tuned.validationMetrics[best_grid_idx]), 5))

    best_name = name + "_optimizado"
    trained_models[best_name] = tuned.bestModel
    metrics_tuned = evaluate_predictions(tuned.bestModel.transform(val))
    model_results.append({"modelo": best_name, "segundos": opt_seconds,
                          **metrics_tuned})
    print(f"AUC-PR en validación externa: {metrics_tuned['auc_pr']:.5f}")

print("\nOptimización de Random Forest y GBT finalizada.")



results = pd.DataFrame(model_results).sort_values("auc_pr", ascending=False).reset_index(drop=True)
columns_to_show = ["modelo", "auc_pr", "auc_roc", "precision", "recall", "f1", "segundos"]
print(results[columns_to_show].round(4).to_string(index=False))
results.to_csv(OUT / "comparacion_validacion.csv", index=False)



comparison_features = None
if COMPARAR_SALDOS_POSTERIORES and POST:
    print("Comparando variables anteriores frente a anteriores + posteriores...")
    retrospective = Pipeline(
        stages=make_preprocessing(NUM + POST, CAT) + [LogisticRegression(**lr_params)]
    ).fit(train)
    retrospective_metrics = evaluate_predictions(retrospective.transform(val))
    operational_metrics = next(r for r in model_results if r["modelo"] == "RegresionLogistica")

    comparison_features = pd.DataFrame([
        {"variante": "Operativo (antes)", "n_variables": len(NUM)+len(CAT),
         "auc_pr": operational_metrics["auc_pr"], "precision": operational_metrics["precision"],
         "recall": operational_metrics["recall"]},
        {"variante": "Retrospectivo (antes + después)", "n_variables": len(NUM)+len(POST)+len(CAT),
         "auc_pr": retrospective_metrics["auc_pr"], "precision": retrospective_metrics["precision"],
         "recall": retrospective_metrics["recall"]},
    ])
    print(comparison_features.round(4).to_string(index=False))
    comparison_features.to_csv(OUT / "comparacion_variables.csv", index=False)
    print("Nota: la comparación explora DISPONIBILIDAD TEMPORAL, no causalidad ni solo correlación.")
else:
    print("Experimento opcional omitido. No afecta a la selección del modelo operativo.")



winner = str(results.iloc[0]["modelo"])
best_model = tuned.bestModel if winner == "LR_optimizada" else trained_models[winner]
print("Modelo seleccionado:", winner)
print("AUC-PR de validación:", round(float(results.iloc[0]["auc_pr"]), 5))



score_val = (
    best_model.transform(val)
    .select("label", vector_to_array("probability")[1].alias("score"))
    .persist(StorageLevel.MEMORY_AND_DISK)
)
thresholds = [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20,
              0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

exprs = [F.sum(F.col("label")).alias("total_fraudes")]
for i, threshold in enumerate(thresholds):
    exprs.extend([
        F.sum(F.when((F.col("label") == 1) & (F.col("score") >= threshold), 1)
              .otherwise(0)).alias(f"tp_{i}"),
        F.sum(F.when((F.col("label") == 0) & (F.col("score") >= threshold), 1)
              .otherwise(0)).alias(f"fp_{i}")
    ])
counts = score_val.agg(*exprs).first().asDict()
total_frauds_val = int(counts["total_fraudes"])
threshold_rows = []
for i, threshold in enumerate(thresholds):
    tp = int(counts[f"tp_{i}"] or 0)
    fp = int(counts[f"fp_{i}"] or 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / total_frauds_val if total_frauds_val else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    threshold_rows.append({"umbral": threshold, "precision": prec, "recall": rec,
                           "f1": f1, "fraudes_detectados": tp, "falsas_alarmas": fp})

threshold_table = pd.DataFrame(threshold_rows).sort_values("f1", ascending=False)
chosen_threshold = float(threshold_table.iloc[0]["umbral"])
print(threshold_table.round(4).to_string(index=False))
print("Umbral elegido (F1 máximo):", chosen_threshold)
threshold_table.to_csv(OUT / "umbrales_validacion.csv", index=False)
score_val.unpersist()



score_test = best_model.transform(test)
pred_test = score_test.withColumn(
    "prediction",
    F.when(vector_to_array("probability")[1] >= chosen_threshold, 1.0).otherwise(0.0)
).persist(StorageLevel.MEMORY_AND_DISK)

final = evaluate_predictions(pred_test)
summary = pd.DataFrame([{ "modelo": winner, "umbral": chosen_threshold, **final }])
print(summary[["modelo", "umbral", "auc_pr", "auc_roc", "precision", "recall", "f1"]]
      .round(5).to_string(index=False))
print(f"Fraudes detectados: {final['tp']} | Fraudes no detectados: {final['fn']}")
print(f"Falsas alarmas: {final['fp']} | Normales bien clasificadas: {final['tn']}")
summary.to_csv(OUT / "metricas_test.csv", index=False)

conf_matrix = np.array([[final["tn"], final["fp"]],
                        [final["fn"], final["tp"]]])
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.imshow(conf_matrix, cmap="Blues")
for row in range(2):
    for col in range(2):
        ax.text(col, row, f"{conf_matrix[row, col]:,}",
                ha="center", va="center", fontsize=10)
ax.set_xticks([0, 1], ["Normal", "Fraude"])
ax.set_yticks([0, 1], ["Normal", "Fraude"])
ax.set_xlabel("Predicción")
ax.set_ylabel("Clase real")
ax.set_title("Matriz de confusión en test")
plt.tight_layout()
plt.savefig(OUT / "03_matriz_confusion_test.png", dpi=140)
plt.show()
pred_test.unpersist()



rf_model = trained_models["RandomForest"]
rf_importance = rf_model.stages[-1].featureImportances.toArray()
feature_meta = rf_model.transform(val.limit(1)).schema["features"].metadata
attrs = feature_meta.get("ml_attr", {}).get("attrs", {})
feature_names = {
    int(item["idx"]): item["name"]
    for attr_group in attrs.values() for item in attr_group
}
importance_table = pd.DataFrame([
    {"variable": feature_names.get(i, f"feature_{i}"), "importancia": float(value)}
    for i, value in enumerate(rf_importance)
]).sort_values("importancia", ascending=False)
print(importance_table.head(8).round(4).to_string(index=False))
importance_table.to_csv(OUT / "importancia_variables_rf.csv", index=False)


print("=== CONCLUSIONES AUTOMÁTICAS ===")
print(f"El conjunto contiene {N:,} transacciones; el fraude representa {prevalence:.4%}.")
print(f"Mejor modelo operativo en validación (AUC-PR): {winner}.")
print(f"Umbral seleccionado sin usar test: {chosen_threshold}.")
print(f"En test: AUC-PR={final['auc_pr']:.4f}, precisión={final['precision']:.4f}, "
      f"recall={final['recall']:.4f}, F1={final['f1']:.4f}.")
print(f"Detecta {final['tp']} fraudes; omite {final['fn']} y genera {final['fp']} falsas alarmas.")
if comparison_features is not None:
    print("La comparación con saldos posteriores es EXPLORATORIA y no valida su uso en tiempo real.")
print("Limitaciones: datos históricos/simulados según la fuente, partición aleatoria, "
      "posibles cambios temporales y costes reales de los errores no modelados.")



result_info = {
    "filas": int(N), "fraude_porcentaje": 100 * float(prevalence),
    "modelo": winner, "umbral": chosen_threshold,
    "metricas_test": {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                      for k, v in final.items()},
    "variables_operativas": NUM + CAT,
    "variables_retrospectivas_solo_experimento": POST,
}
with open(OUT / "resultados_finales.json", "w", encoding="utf-8") as fh:
    json.dump(result_info, fh, ensure_ascii=False, indent=2)
print("Resumen guardado:", OUT / "resultados_finales.json")

if GUARDAR_MODELO:
    model_path = OUT / "modelo_final_spark"
    try:
        best_model.write().overwrite().save(str(model_path.resolve()))
        print("Modelo guardado en:", model_path.resolve())
    except Exception as exc:
        print("No se ha podido exportar el modelo Spark:", str(exc)[:300])
        print("Las métricas y tablas sí están guardadas; verifica Spark/Hadoop en tu entorno.")



print("Spark master:", spark.sparkContext.master)
print("Particiones de train:", train.rdd.getNumPartitions())
print("Particiones de val:", val.rdd.getNumPartitions())

EJECUTAR_EJEMPLO_RDD = False
if EJECUTAR_EJEMPLO_RDD:
    rdd_counts = (
        train.select("label").rdd
        .map(lambda row: (int(row["label"]), 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )
    print("Recuento de clases calculado con RDD:", sorted(rdd_counts))

train.unpersist()
val.unpersist()
test.unpersist()
print("Análisis finalizado. Revisa la carpeta:", OUT.resolve())