from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            AREA=request.form.get('AREA'),
            INT_SQFT = float(request.form.get('INT_SQFT')),
            N_BEDROOM = float(request.form.get('N_BEDROOM')),
            N_BATHROOM = float(request.form.get('N_BATHROOM')),
            N_ROOM = float(request.form.get('N_ROOM')),
            SALE_COND = request.form.get('SALE_COND'),
            PARK_FACIL = request.form.get('PARK_FACIL'),
            UTILITY_AVAIL= request.form.get('UTILITY_AVAIL'),
            STREET = request.form.get('STREET'),
            MZZONE = request.form.get('MZZONE'),
            BUILDTYPE = request.form.get('BUILDTYPE')        

        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)