from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import logging

logging.basicConfig(filename="log.txt",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)

# Loading Pickle File
xgb = pickle.load(open("XGBModel.pkl", "rb"))

# Data Processing Function
def process_data(data):
    """ This Function Take data and remove its Skewness for
        Model Prediction """
    try:
        val1 = data[0]**3
        val2 = np.log1p(data[1])
        val3 = np.log1p(data[2])
        val4 = np.log1p(data[3])

        logging.info("Right and Left skew data are Adjusted.")
        data1 = list((val1, val2, val3, val4))

        return data1
    except Exception as e:
        logging.error(e)
        print("Error in Preprocessing")

# Route URL
@app.route('/', methods=['GET', 'POST'])
def home():
    # Set Index HTML
    return render_template('home.html')

# Prediction URL
@app.route('/predict', methods=['POST'])
def Form_page():
    try:
        if (request.method == 'POST'):

            ISI = float(request.form.get("ISI"))
            BUI = float(request.form.get("BUI"))
            FWI = float(request.form.get("FWI"))
            FFMC = float(request.form.get("FFMC"))

        val = list((FFMC, ISI, BUI, FWI))
        print("data Collected : ", val)

        # Processing Right and Left Skew Data list
        val = [process_data(val)]
        print("Processed", val)

        # Predicting Output
        Answer = xgb.predict(val)[0]
        Answer = "Fire" if Answer == 1 else "No Fire"
        print("Pridicted Output is : ", Answer)

        return render_template("result.html", Answer=Answer)

    except Exception as e:
        print("Form Page problem")
        logging.error(e)
        return render_template('home.html')


@app.route('/predict_api', methods=["GET", "POST"])
def Api_test():
    try:
        logging.info("Requested Through API")
        if request.method == "POST":
            ISI = float(request.form.get("ISI"))
            BUI = float(request.form.get("BUI"))
            FWI = float(request.form.get("FWI"))
            FFMC = float(request.form.get("FFMC"))

            val = list((FFMC, ISI, BUI, FWI))
            print("data Collected : ", val)

            val = [process_data(val)]
            print("Processed", val)

            # Predicting Output
            Answer = xgb.predict(val)[0]
            Answer = "Fire" if Answer == 1 else "No Fire"
            print("Pridicted Output is : ", Answer)

            logging.info("Prediction Return in Json Format")
            return jsonify({"Output": Answer})
    except Exception as e:
        logging.error(e)
        return jsonify({"Response": "Bad Request"})


if __name__ == "__main__":
    app.run(debug=False)