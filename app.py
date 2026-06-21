from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargo el pipeline completo (codificación + escalado + PCA + modelo)
pipeline = joblib.load("modelo_bike_sharing.pkl")

# Mapas de las variables categóricas, para mostrar etiquetas comprensibles en el formulario
MAPA_SEASON = {"1": 1, "2": 2, "3": 3, "4": 4}  # 1=Primavera, 2=Verano, 3=Otoño, 4=Invierno
MAPA_WEATHERSIT = {"1": 1, "2": 2, "3": 3, "4": 4}  # 1=Despejado ... 4=Lluvia/nieve fuerte
MAPA_WEEKDAY = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
MAPA_YR = {"2011": 0, "2012": 1}


def normalizar_temp(temp_celsius):
    return (temp_celsius + 8) / 47


def normalizar_atemp(atemp_celsius):
    return (atemp_celsius + 16) / 66


def normalizar_hum(hum_porcentaje):
    return hum_porcentaje / 100


def normalizar_windspeed(windspeed_real):
    return windspeed_real / 67


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    error = None

    if request.method == "POST":
        try:
            # Recibo y valido los datos del formulario
            anio = request.form["anio"]
            mes = int(request.form["mes"])
            hora = int(request.form["hora"])
            dia_semana = request.form["dia_semana"]
            es_festivo = int(request.form["es_festivo"])
            es_laboral = int(request.form["es_laboral"])
            estacion = request.form["estacion"]
            clima = request.form["clima"]
            temp_real = float(request.form["temperatura"])
            atemp_real = float(request.form["sensacion_termica"])
            hum_real = float(request.form["humedad"])
            wind_real = float(request.form["viento"])

            # Validación de rangos razonables
            if not (1 <= mes <= 12):
                raise ValueError("El mes debe estar entre 1 y 12.")
            if not (0 <= hora <= 23):
                raise ValueError("La hora debe estar entre 0 y 23.")
            if not (-30 <= temp_real <= 50):
                raise ValueError("La temperatura debe estar entre -30°C y 50°C.")
            if not (-30 <= atemp_real <= 60):
                raise ValueError("La sensación térmica debe estar entre -30°C y 60°C.")
            if not (0 <= hum_real <= 100):
                raise ValueError("La humedad debe estar entre 0% y 100%.")
            if not (0 <= wind_real <= 100):
                raise ValueError("La velocidad del viento debe ser un valor positivo razonable.")

            # Construyo el registro con las mismas 12 columnas usadas en el entrenamiento
            datos = pd.DataFrame([{
                "season": MAPA_SEASON[estacion],
                "yr": MAPA_YR[anio],
                "mnth": mes,
                "hr": hora,
                "holiday": es_festivo,
                "weekday": MAPA_WEEKDAY[dia_semana],
                "workingday": es_laboral,
                "weathersit": MAPA_WEATHERSIT[clima],
                "temp": normalizar_temp(temp_real),
                "atemp": normalizar_atemp(atemp_real),
                "hum": normalizar_hum(hum_real),
                "windspeed": normalizar_windspeed(wind_real),
            }])

            prediccion = pipeline.predict(datos)[0]
            resultado = round(prediccion)

        except (ValueError, KeyError) as e:
            error = f"Datos inválidos: {e}"
        except Exception as e:
            error = f"Ocurrió un error al procesar la predicción: {e}"

    return render_template("index.html", resultado=resultado, error=error)


if __name__ == "__main__":
    app.run(debug=True)